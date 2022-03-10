# Elastic Weight Consolidation in PyTorch

This is an implementation of Elastic Weight Consolidation (EWC)<sup>[1](#ewc)</sup> in [PyTorch](https://pytorch.org)

## Example

```python
import torch.nn.functional as F
from ewc import EWC

module = ...
samples = ...

f = lambda m, s: F.log_softmax(m(s))

ewc = EWC(module, samples, f)

# Code that changes module

l = ewc.loss(module)
```

## Acknowledgements

The method for calculating the jacobian with respect to the models parameters was based on [NNGeometry](https://github.com/tfjgeorge/nngeometry).

<a name="ewc">1</a>: James Kirkpatrick, et al. "Overcoming catastrophic forgetting in neural networks". CoRR abs/1612.00796. (2016).