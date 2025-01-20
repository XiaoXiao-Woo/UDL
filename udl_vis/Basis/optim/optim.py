import numpy as np
import torch
from torch import nn
import torch.optim as optim


def get_optimizer(model: torch.nn.Module, params: "Iterable | dict", **kwargs):
    name = kwargs.pop("optimizer_type").lower()
    if name == "sgd":
        return optim.SGD(params, **kwargs)
    elif name == "adam":
        return optim.Adam(params, **kwargs)
    elif name == "adamw":
        return optim.AdamW(params, **kwargs)
    elif name == 'lion':
        from lion_pytorch import Lion
        return Lion(params, betas=(0.95, 0.98), use_triton=True, **kwargs) 
    elif name == 'fusedadam':
        import deepspeed
        return deepspeed.ops.adam.FusedAdam(params, **kwargs)
    elif name == 'schedulefree_adam':
        import schedulefree
        return schedulefree.AdamWScheduleFree(params, **kwargs)
    elif name == 'adam_mini':
        import Adam_mini
        return Adam_mini(model.named_parameters(), **kwargs)
    elif name == 'adamw_8bit':
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(params, **kwargs)
    elif name == 'ademamix':
        import AdEMAMix
        return AdEMAMix(params, **kwargs)
    elif name == 'shampoo':
        from torch_optimizer import Shampoo
        return Shampoo(params, **kwargs)
    elif name == 'shampoo_ddp':
        import warnings
        from accelerate import PartialState, DistributedType
        from functools import partial

        from shampoo_optimizers.distributed_shampoo.distributed_shampoo import (
            DistributedShampoo, AdamGraftingConfig, DDPShampooConfig
        )
        from shampoo_optimizers.distributed_shampoo.utils.shampoo_ddp_distributor import CommunicationDType

        warnings.warn('Shampoo optimizer has not been tested yet, may cause nan or other unexpected errors.', UserWarning)

        state = PartialState()

        if state.distributed_type == DistributedType.MULTI_GPU:
            distributed_adam_config=DDPShampooConfig(
                communication_dtype=CommunicationDType.FP32,
                num_trainers_per_group=state.num_processes,
                communicate_params=False,
            )
        elif state.distributed_type == DistributedType.NO:
            distributed_adam_config = None
        else:
            raise ValueError(f'Shampoo optimizer only supports DDP and NO distributed type, but got {state.distributed_type}')

        opt = DistributedShampoo(
            params,
            lr=kwargs['lr'],
            betas=kwargs.pop('betas', (0.9, 0.999)),
            epsilon=kwargs.pop('eps', 1e-12),
            weight_decay=kwargs.pop('weight_decay', 1e-5),
            max_preconditioner_dim=8192,
            precondition_frequency=100,
            use_nesterov=kwargs.pop('use_nesterov', False),
            use_pytorch_compile=kwargs.pop('use_pytorch_compile', True),
            use_decoupled_weight_decay=True,
            grafting_config=AdamGraftingConfig(
                beta2=kwargs.pop('beta2', 0.999),
                epsilon=kwargs.pop('eps', 1e-12),
            ),
            distributed_config=distributed_adam_config,
        )
        opt.state_dict = partial(opt.distributed_state_dict, key_to_param=model.named_parameters())
        opt.load_state_dict = partial(opt.load_distributed_state_dict, key_to_param=model.named_parameters())

        return opt
    elif name == 'soap':
        from .SOAP import SOAP
        return SOAP(params, **kwargs)
    else:
        raise NotImplementedError(f'optimizer {name} not implemented')


def partial_train(model, layers: list):
    # forzen layers
    for param in model.parameters():
        if layers is not None and layers in param:
            continue
        param.requires_grad = False

    # Replace the last fc layer
    model.fc = nn.Linear(512, 100)
    return model
