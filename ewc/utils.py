# Deletes a parameter from a torch.Module based on the 
# name given by the named_parameters function
def delparam(module, name):
    name_list = name.split('.')
    for name in name_list[:-1]:
        module = getattr(module, name)
    
    delattr(module, name_list[-1])


# Sets a parameter to a torch.Module given a name
# in the format of generated by named_parameters
def setparam(module, name, param):
    name_list = name.split('.')
    for name in name_list[:-1]:
        module = getattr(module, name)

    setattr(module, name_list[-1], param)