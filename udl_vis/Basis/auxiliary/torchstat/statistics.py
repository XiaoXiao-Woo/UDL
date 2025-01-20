import warnings

import torch
import torch.nn as nn
from . import ModelHook
from collections import OrderedDict
from . import StatTree, StatNode, report_format


def get_parent_node(root_node, stat_node_name):
    assert isinstance(root_node, StatNode)

    node = root_node
    names = stat_node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i+1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_stat_tree(leaf_modules):
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = StatNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        # if 'HSI_Fusion' == leaf_module_name:
        #     print("111", leaf_module_name, leaf_module.__class__.__name__)
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            create_index += 1
            stat_node_name = '.'.join(names[0:i+1])
            parent_node = get_parent_node(root_node, stat_node_name)
            node = StatNode(name=stat_node_name, mtype=leaf_module.__base__ if hasattr(leaf_module, '__base__') else leaf_module.__class__.__name__, parent=parent_node)#.__class__.__name__
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                try:
                    input_shape = leaf_module.input_shape.numpy().tolist()
                    output_shape = leaf_module.output_shape.numpy().tolist()
                    node.input_shape = input_shape
                    node.output_shape = output_shape
                    node.parameter_quantity = leaf_module.parameter_quantity.numpy()[0]
                    node.inference_memory = leaf_module.inference_memory.numpy()[0]
                    node.MAdd = leaf_module.MAdd.numpy()[0]
                    node.Flops = leaf_module.Flops.numpy()[0]
                    node.duration = leaf_module.duration.numpy()[0]
                    node.Memory = leaf_module.Memory.numpy().tolist()
                except AttributeError as e:
                    print(names, leaf_module)
                    raise AttributeError(e)
    return StatTree(root_node)


class ModelStat(object):
    def __init__(self, model, input_size, query_granularity=1, device="cuda", keep_pair=False, ignore_flops=False): # , debug_layers=[]
        assert isinstance(model, nn.Module)
        # assert isinstance(input_size, (tuple, list)) and len(input_size) == 3
        self._model = model
        self._input_size = input_size
        self._query_granularity = query_granularity
        # self.debug_layers = debug_layers
        self.keep_pair = keep_pair
        self.device = device
        self.ignore_flops = ignore_flops

    def _analyze_model(self):
        model_hook = ModelHook(self._model, self._input_size, self.device, self.keep_pair, self.ignore_flops) # , debug_layers=self.debug_layers
        leaf_modules = model_hook.retrieve_leaf_modules()
        stat_tree = convert_leaf_modules_to_stat_tree(leaf_modules)
        collected_nodes = stat_tree.get_collected_stat_nodes(self._query_granularity) # self.debug_layers,
        model_hook.clear_hooks()
        return collected_nodes

    def show_report(self):
        collected_nodes = self._analyze_model()
        report = report_format(collected_nodes)
        print(report)

def stat(model, input_size, query_granularity=1, device="cuda", keep_pair=False, ignore_flops=False):#, debug_layers=["MSA", "SwinTEB", "XCTEB", "MSA_BNC", 'cGCN', 'sGCN']):
    warnings.warn("Note that for LayerNorm, the function name uses the full name")
    ms = ModelStat(model, input_size, query_granularity, device, keep_pair, ignore_flops) #debug_layers
    ms.show_report()
