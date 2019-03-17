from torch.nn import Module, DataParallel
import torch

def getParallelModel(arch:Module, device_ids = [0, 1]):
    arch = DataParallel(module=arch, device_ids=device_ids)
    arch = arch.cuda()
    return arch


def getCUDAModel(arch:Module):
    arch = arch.cuda()
    return arch


def unpackParallel(baseModel:Module, arch_loc:str, device_ids = [0, 1]):
    model = baseModel
    model = DataParallel(module=model, device_ids=device_ids)
    model.load_state_dict(torch.load(arch_loc))
    model = model.module

    return model