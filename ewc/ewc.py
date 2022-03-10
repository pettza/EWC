import copy
from functools import partial
from typing import Iterable, Union, Callable, Generator

import torch
from torch.autograd.functional import jacobian
from torch.nn import Linear

from .module_parameters import ModuleParameters
from .utils import delparam, setparam


class EWC:
    '''Implements Elastic Weight Consolidations'''

    def __init__(self,
                 module: torch.nn.Module,
                 samples: Union[torch.Tensor, Iterable[torch.Tensor]],
                 fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
                 method='torch'):
        '''
        Args:
            module : The torch module
            samples: A tensor with the samples to use for the Fisher diagonal estimation
                     or an iterable of tensors for batched estimation
            fn : A fnction that takes a module and a tensor and returns the likelihood
            method: The method for computing the jacobian wrt the parametes. 
                    Choices are
                        - 'torch', uses the the jacobian function of PyTorch.
                        This method is slow because gradient is called for each sample,
                        but it works for any module
                        - 'hooks', uses backward hooks. This method is very fast, but
                        this code has support only for linear layer, although support
                        for other layers can be added by editing the _compute_kernel_diag_hooks
                        member function and implementing different hooks
        '''
        self.module = copy.deepcopy(module) # We need to change the module so make a deepcopy of it
        self.mean_params = ModuleParameters(self.module)

        if method == 'torch':
            self.kernel_diag = self._compute_kernel_diag_torch(samples, fn)
        elif method == 'hooks':
            self.kernel_diag = self._compute_kernel_diag_hooks(samples, fn)
        else:
            raise NotImplementedError(f'Invalid method: {method}')
    
    def loss(self, other_module):
        '''Computes the EWC loss'''
        other_params = ModuleParameters(other_module)

        return 0.5 * self.kernel_diag.dot((self.mean_params - other_params)**2)
        
    def _compute_kernel_diag_torch(self, samples, fn):
        self.module.eval()
        
        # Get list of names and parameters
        names, params = list(zip(*self.module.named_parameters()))

        # PyTorch jacobian function copies the tensor objects
        # so we need a function that takes the parameters and
        # uses them so that they are part of the computation graph
        def jac_f(samples, *params):
            # This process doesn't work if the attributes are torch.nn.Parameters
            # so set them as torch.Tensor. This potentially breaks code, although
            # it seems unlikely
            for n, p in zip(names, params):
                delparam(self.module, n)
                setparam(self.module, n, p)

            output = fn(self.module, samples)
            return output.squeeze(-1)


        if isinstance(samples, torch.Tensor):
            jac = jacobian(lambda *params: jac_f(samples, *params), params)
            
            kernel_diag = ModuleParameters.from_dict(
                {n: (p**2).mean(dim=0) for n, p in zip(names, jac)})
        else:
            size = 0
            for i, batch in enumerate(samples):
                size += batch.shape[0]

                jac = jacobian(lambda *params: jac_f(batch, *params), params)
                
                if i == 0:
                    kernel_diag = ModuleParameters.from_dict(
                        {n: (p**2).sum(dim=0) for n, p in zip(names, jac)})
                else:
                    kernel_diag += ModuleParameters.from_dict(
                        {n: (p**2).sum(dim=0) for n, p in zip(names, jac)})
            
            kernel_diag /= size

        return kernel_diag

    def _compute_kernel_diag_hooks(self, samples, fn):
        self.module.eval()

        handles = []
        for m in self.module.modules():
            if isinstance(m, Linear):
                hf = m.register_forward_hook(self._forward_hook)
                hb = m.register_full_backward_hook(self._backward_hook)
                handles.extend((hf, hb))
        
        self.saved = dict()
        self.grads = dict()
        self.batched = not isinstance(samples, torch.Tensor)

        if not self.batched:
            samples = samples.detach() # Make shallow copy of samples
            samples.requires_grad = True # This is needed in order for the backward 
                                         # computation to propagate beyond the last layer
            out = fn(self.module, samples)
            out.sum(dim=0).backward()
            
            kernel_diag = ModuleParameters.from_dict(
                {n: self.grads[p] for n, p in self.module.named_parameters()})
        else:
            size = 0
            for i, batch in enumerate(samples):
                size += batch.shape[0]

                batch.requires_grad = True # This is needed in order for the backward 
                                           # computation to propagate beyond the last layer
                out = fn(self.module, batch)
                out.sum(dim=0).backward()

                if i == 0:
                    grads_acc = self.grads
                    self.grads = dict()
                else:
                    for k in grads_acc.keys():
                        grads_acc[k] += self.grads[k]
            
            kernel_diag = ModuleParameters.from_dict(
                {n: grads_acc[p] for n, p in self.module.named_parameters()})
            kernel_diag /= size
        
        del self.saved
        del self.grads
        del self.batched

        for h in handles:
            h.remove()

        return kernel_diag

    def _forward_hook(self, mod, input, output):
        self.saved[mod] = input[0] # input is always a tuple

    def _backward_hook(self, mod, grad_input, grad_output):
        x = self.saved[mod]
        gy = grad_output[0] # grad_output is always a tuple
        t = torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)) ** 2
        self.grads[mod.weight] = torch.sum(t, dim=0) if self.batched else torch.mean(t, dim=0)
        if mod.bias is not None:
            self.grads[mod.bias] = torch.sum(gy**2, dim=0) if self.batched else torch.mean(gy**2, dim=0)
