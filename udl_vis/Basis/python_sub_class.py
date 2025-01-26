# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:

from udl_vis.Basis.config import Config
import warnings
import os
import sys
import importlib
import copy

# import scipy.io as sio


class TaskDispatcher(Config):
    _task = dict()

    def __init_subclass__(cls, name="", **kwargs):
        super().__init_subclass__(**kwargs)

        if name != "":
            cls._task[name] = cls
            cls._name = name
            # print(cls.__repr__, cls..__repr__)
        else:
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
            cls._task[cls.__name__] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        if cls is TaskDispatcher:
            task = kwargs.get("task")
            try:
                cls = cls._task[task]
            except KeyError:
                raise ValueError(
                    f"Got task={task} but expected" f"one of {cls._task.keys()}"
                )

        instance = super().__new__(cls)

        return instance

    # def __len__(self):
    #     return len(self._cfg_dict)
    #
    # def __getattr__(self, name):
    #     return getattr(self._cfg_dict, name)
    #
    # def __delattr__(self, name):
    #     return delattr(self._cfg_dict, name)
    #
    # def __getitem__(self, name):
    #     return self._cfg_dict.__getitem__(name)
    #
    # def __iter__(self):
    #     return iter(self._cfg_dict)
    #
    # def __repr__(self):
    #     return f'TaskDispatcher {self._cfg_dict.__repr__()}'

    # def __setattr__(self, name, value):
    #     if isinstance(value, dict):
    #         value = ConfigDict(value)
    #     print("__setattr__")
    #     self._cfg_dict.__setattr__(name, value)

    # def __setitem__(self, name, value):
    #     if isinstance(value, dict):
    #         value = ConfigDict(value)
    #     print("__setitem__")
    #     self._cfg_dict.__setitem__(name, value)

    @classmethod
    def new(cls, **kwargs):
        # 需要从外部启动和从任务启动，但参数不同
        key = "mode"
        value = kwargs.setdefault("mode", None)
        print(f'111, set mode="{value}" from [entrypoint, nni]')
        if value is None:
            # 第二、三调用层进入此函数
            key = "task"
            if kwargs.get("task", None):
                # 二
                value = kwargs.pop("task")
                print("222", value)
            elif kwargs.get("arch", None):
                # 三
                key = "arch"
                value = kwargs.pop("arch")
                print("333", value)
            else:
                key = "arch"

        kwargs.pop("mode")

        try:
            cls = cls._task[value]
        except KeyError:
            warning = f"Got {key}={value} but expected " f"one of {cls._task.keys()}"
            warnings.warn(warning)
            return Config()

        return cls(**kwargs)


# class ModelDispatcher(object):
#     _task = dict()
#
#     def __init_subclass__(cls, name='', **kwargs):
#         super().__init_subclass__(**kwargs)
#         if name != '':
#             cls._task[name] = cls
#             cls._name = name
#             # print(cls.__repr__, cls..__repr__)
#         else:
#             # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
#             cls._task[cls.__name__] = cls
#             cls._name = cls.__name__
#
#     def __new__(cls, *args, **kwargs):
#         if cls is ModelDispatcher:
#             task = kwargs.get('task')
#             try:
#                 cls = cls._task[task]
#             except KeyError:
#                 raise ValueError(f'Got task={task} but expected'
#                                  f'one of {cls._task.keys()}')
#
#         instance = super().__new__(cls)
#
#         return instance
#
#     @classmethod
#     def build_model(cls, cfg):
#
#         arch = cfg.arch
#         task = cfg.task
#         model_style = cfg.model_style
#
#         try:
#             # 获得PansharpeningModel,进行分发
#             cls = cls._task[task](None, None)
#         except KeyError:
#             raise ValueError(f'Got task={task} but expected '
#                              f'one of {cls._task.keys()} in {cls}')
#         try:
#             # 获得具体的模型
#             cls_arch = cls._models[arch]()
#         except KeyError:
#             raise ValueError(f'Got arch={arch} but expected '
#                              f'one of {cls._models.keys()} in {cls}')
#
#         model, criterion, optimizer, scheduler = cls_arch(cfg)
#
#         if model_style is None:
#             # 获得PansharpeningModel,model+head
#             model_style = task
#
#         if model_style is not None:
#             try:
#                 # 获得具体的模型
#                 model = cls._task[model_style](model, criterion)
#             except KeyError:
#                 raise ValueError(f'Got model_style={model_style} but expected '
#                                  f'one of {cls._models.keys()} (merged in _models) in {cls}')
#
#         return model, criterion, optimizer, scheduler
#


class ModelDispatcher(object):
    _task = dict()
    # __init_subclass__调用优先级高于__init__, 无法使用self._task,可以使用cls._task

    @classmethod
    def create_fusion_model(cls, name, *base_classes):
        return type(f"{name}", base_classes, {})

    def __init_subclass__(cls, name="", **kwargs):
        super().__init_subclass__(**kwargs)
        if name != "":
            if isinstance(name, list):
                for index, value in enumerate(name):
                    if index == 0:
                        cls._name = value
                    fusion_model_class = cls.create_fusion_model(value, cls)
                    cls._task[value] = fusion_model_class
                    fusion_model_class._name = value
            else:
                cls._task[name] = cls
                cls._name = name
        else:
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
            cls._task[cls.__name__] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        if cls is ModelDispatcher:
            task = kwargs.get("task")
            try:
                cls = cls._task[task]
            except KeyError:
                raise ValueError(
                    f"Got task={task} but expected" f"one of {cls._task.keys()}"
                )

        instance = super().__new__(cls)

        return instance

    @classmethod
    def build_model_from_task(cls, cfg, logger):
        # TODO: baseline + head, structure like DETR/mmlab,
        #  but you hardly direct know about the model structure
        arch = cfg.model_type
        task = cfg.task
        # model_style = cfg.model_style
        try:
            # 获得PansharpeningModel,进行分发
            cls = cls._task[task](device=None, model=None, criterion=None, logger=None)
        except KeyError:
            raise ValueError(
                f"Got task={task} but expected " f"one of {cls._task.keys()} in {cls}"
            )
        
        if cfg.get("import_path", None) is not None:
            if os.path.isfile("/".join([*cfg.import_path, "__init__.py"])):
                sys.path.append(cfg.import_path[0])
                importlib.import_module(cfg.import_path[1], "__init__")
        else:
            print(f"Only load models from {cls}: {cls._models.keys()} for {cfg.task}")
        try:
            # The following code replaces the `register` function to obtain the specific model
            # Logic:
            # cls is parent class
            # class build_model(cls) is the specific model
            # Thus, we can obtain the specific model
            cls_arch = cls._models[arch](device=None)
        except KeyError:
            if cfg.get("import_path", None) is not None:
                raise ValueError(
                    f"Got arch={arch} but expected "
                    f"one of {cls._models.keys()} in {cls}, "
                    f"because import_path: {"/".join([*cfg.import_path, "__init__.py"])} is wrong."
                )
            else:
                raise ValueError(
                    f"Got arch={arch} but expected "
                    f"one of {cls._models.keys()} in {cls}"
                )

        model, criterion, optimizer, scheduler = cls_arch(cfg, logger)

        # if model_style is None:
        #     # 获得PansharpeningModel,model+head
        #     model_style = task

        # if model_style is not None:
        #     try:
        #         # 获得具体的模型
        #         model = cls._task[model_style](cfg.device, model, criterion)
        #     except KeyError:
        #         raise ValueError(
        #             f"Got model_style={model_style} but expected "
        #             f"one of {cls._models.keys()} (merged in _models) in {cls}"
        #         )

        return model, criterion, optimizer, scheduler

    @classmethod
    def build_task_from_model(cls, cfg):

        arch = cfg.arch

        try:
            # 获得具体的模型
            cls_arch = cls._task[arch]()
        except KeyError:
            raise ValueError(
                f"Got arch={arch} but expected " f"one of {cls._task.keys()} in {cls}"
            )

        model, criterion, optimizer, scheduler = cls_arch(cfg)

        return model, criterion, optimizer, scheduler
