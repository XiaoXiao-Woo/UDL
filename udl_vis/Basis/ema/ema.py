# Copyright (c) OpenMMLab. All rights reserved.
import torch
from udl_vis.Basis.hook import Hook
from udl_vis.mmcv.utils import print_log
import copy


class EMAHook(Hook):
    r"""Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointSaverHook.

        .. math::

            Xema\_{t+1} = (1 - \text{momentum}) \times
            Xema\_{t} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Defaults to 100.
        resume_from (str): The checkpoint path. Defaults to None.
    """

    def __init__(
        self,
        model,
        momentum=0.0002,
        interval=1,
        warm_up=100,
        resume_from=None,
        state_include_online_model=False,
        **load_model_func_kwargs,
    ):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert momentum > 0 and momentum < 1
        self.momentum = momentum**interval
        self.checkpoint = resume_from
        self.model = model
        self.state_include_online_model = state_include_online_model
        self.load_model_func_kwargs = load_model_func_kwargs

    def before_run(self):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        # model = extract_model_from_parallel(self.model)
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
        self.model_parameters = dict(model.named_parameters(recurse=True))
        # for name, value in self.model_parameters.items():
        #     # "." is not allowed in module's buffer name
        #     buffer_name = f"ema_{name.replace('.', '_')}"
        #     self.param_ema_buffer[name] = buffer_name
        #     model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.detach_()

        self.param_ema_buffer = dict(self.ema_model.named_buffers(recurse=True))

        # self.ema_model = model.__class__(**model.__dict__)  # 创建新实例
        # self.ema_model.load_state_dict(model.state_dict())
        # if self.checkpoint is not None:
        #     runner.resume(self.checkpoint)

    """
        def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)

    """

    def after_train_iter(self, curr_step):
        """Update ema parameter every self.interval iterations."""
        # We warm up the momentum considering the instability at beginning
        momentum = min(self.momentum, (1 + curr_step) / (self.warm_up + curr_step))
        if curr_step % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            buffer_name = self.param_ema_buffer[name]
            buffer_parameter = self.model_buffers[buffer_name]
            buffer_parameter.mul_(1 - momentum).add_(parameter.data, alpha=momentum)

    def after_train_epoch(self):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_train_epoch(self):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)

    ## for accelerator state loading
    def state_dict(self):
        if self.state_include_online_model:
            return {
                "ema_model": self.model_buffers,
                "online_model": self.model_parameters,
            }
        else:
            return {"ema_model": self.model_buffers}

    def load_state_dict(self, state_dict):
        # saved online_model and ema_model
        print_log("load ema_model state dict")

        if (
            self.state_include_online_model
            and state_dict.get("online_model", None) is not None
        ):
            if hasattr(self.online_model, "module"):
                self.ema_model.module.load_state_dict(
                    state_dict["ema_model"], **self.load_model_func_kwargs
                )
                self.online_model.module.load_state_dict(
                    state_dict["online_model"], **self.load_model_func_kwargs
                )
            else:
                self.online_model.load_state_dict(
                    state_dict["online_model"], **self.load_model_func_kwargs
                )
                self.ema_model.load_state_dict(
                    state_dict["ema_model"], **self.load_model_func_kwargs
                )
            print_log("load online_model and ema_model state dict")
        # only has ema_model
        else:
            if hasattr(self.model_parameters, "module"):
                self.ema_model.module.load_state_dict(state_dict, **self.load_model_func_kwargs)
            else:
                self.ema_model.load_state_dict(
                    state_dict["ema_model"], **self.load_model_func_kwargs
                )
            print_log(
                "No online_model is loaded, please make sure you have set `online_model`!"
            )
