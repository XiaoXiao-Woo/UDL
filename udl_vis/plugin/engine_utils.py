import inspect
import udl_vis
from functools import partial


def simple_dispatch_model(cfg, build_model, parser_output=False):

    if parser_output:
        model, criterion, optimizer, scheduler = build_model()
        dispatcher = model()
        return model, criterion, optimizer, scheduler
    else:
        dispatcher = build_model()
        return dispatcher

    # function_signature = inspect.signature(build_model)
    # parameters = function_signature.parameters
    # if len(parameters) == 3:
    #     # hydra_run(build_model=build_pan)
    #     if not parser_output:
    #         return build_model()
    #     else:
    #         model, criterion, optimizer, scheduler = build_model(cfg.arch, cfg.task, cfg)
    #         dispatcher = model
    # else:
    #     # model, criterion, optimizer, scheduler = build_model()(cfg)
    #     dispatcher = build_model(cfg.device)
    #     if not parser_output:
    #         return dispatcher

    #     model, criterion, optimizer, scheduler = dispatcher(cfg)
    #     # When build_model is ModelDispatcher, instead of build_model is TaskDispatcher
    #     if isinstance(dispatcher, udl_vis.Basis.ModelDispatcher):
    #         model = build_model(model, criterion)
    # TODO: deepxde
    # attributes_and_values = {
    #     k: v
    #     for k, v in vars(dispatcher).items()
    #     if k not in ["model", "criterion", "reg"]
    # }
    # [setattr(model, k, v) for k, v in attributes_and_values.items()]
    # return model, criterion, optimizer, scheduler
