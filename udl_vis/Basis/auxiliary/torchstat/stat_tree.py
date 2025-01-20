import queue

import numpy as np


class StatTree(object):
    def __init__(self, root_node):
        assert isinstance(root_node, StatNode)

        self.root_node = root_node

    def get_same_level_max_node_depth(self, query_node):
        if query_node.name == self.root_node.name:
            return 0
        same_level_depth = max([child.depth for child in query_node.parent.children])
        return same_level_depth

    def update_stat_nodes_granularity(self):
        q = queue.Queue()
        q.put(self.root_node)
        while not q.empty():
            node = q.get()
            node.granularity = self.get_same_level_max_node_depth(node)
            for child in node.children:
                q.put(child)

    def get_collected_stat_nodes(self, query_granularity):  # debug_layers
        self.update_stat_nodes_granularity()

        collected_nodes = []
        stack = list()
        stack.append(self.root_node)

        while len(stack) > 0:
            node = stack.pop()

            # if node.mtype == "PanFormerEncoderLayer":
            #     print(node.mtype)
            # if any([L in node.mtype for L in debug_layers]): #node.name
            if node.depth>1:
                node.count_total_params_flops()
            if 'flops' in node.mtype:
                node.mtype = node.mtype.replace("_flops", '')
                # print(node.mtype)
                if node.depth > query_granularity:
                    collected_nodes.append(node)
            # if node.depth > 1:
            #     node.params_proportion = 0
            #     node.Flops_proportion = 0
            for child in reversed(node.children):
                # node.params_proportion += child.parameter_quantity
                # node.Flops_proportion += child.Flops
                stack.append(child)
            if node.depth == query_granularity:
                collected_nodes.append(node)
            if node.depth < query_granularity <= node.granularity:
                collected_nodes.append(node)

        return collected_nodes


class StatNode(object):
    def __init__(self, name=str(), mtype=str(), parent=None):
        self._name = name
        self._mtype = mtype
        self._input_shape = None
        self._output_shape = None
        self._parameter_quantity = 0
        self._inference_memory = 0
        self._MAdd = 0
        self._Memory = (0, 0)
        self._Flops = 0
        self._duration = 0
        self._duration_percent = 0
        self._params_proportion = 0
        self._Flops_proportion = 0

        self._granularity = 1
        self._depth = 1
        self.parent = parent
        self.children = list()

    def count_total_params_flops(self):
        total_params = 0
        total_flops = 0
        stack = []
        stack.append(self)

        if self.parent is None:
            while len(stack) > 0:
                node = stack.pop()
                for child in node.children:
                    stack.append(child)
                    total_params += child._parameter_quantity
                    total_flops += child._Flops
            self.root_total_params = total_params
            self.root_total_flops = total_flops
            self.p_total_params = total_params
            self.p_total_flops = total_flops
        else:
            for child in self.children:
                total_params += child._parameter_quantity
                total_flops += child._Flops
            self.p_total_flops = total_flops
            self.p_total_params = total_params


        for child in self.children:
            child.root_total_flops = self.root_total_flops
            child.root_total_params = self.root_total_params

        # if self.parent is None:
        #     for child in self.children:
        #         child.root_total_flops = total_flops
        #         child.root_total_params = total_params
        # else:
        #     for child in self.children:
        #         child.p_total_flops = total_flops
        #         child.p_total_params = total_params

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def mtype(self):
        return self._mtype

    @mtype.setter
    def mtype(self, mtype):
        self._mtype = mtype

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, g):
        self._granularity = g

    @property
    def depth(self):
        d = self._depth
        if len(self.children) > 0:
            d += max([child.depth for child in self.children])
        return d

    @property
    def input_shape(self):
        if len(self.children) == 0:  # leaf
            return self._input_shape
        else:
            return self.children[0].input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        self._input_shape = input_shape

    @property
    def output_shape(self):
        if len(self.children) == 0:  # leaf
            return self._output_shape
        else:
            return self.children[-1].output_shape

    @output_shape.setter
    def output_shape(self, output_shape):
        assert isinstance(output_shape, (list, tuple))
        self._output_shape = output_shape

    @property
    def parameter_quantity(self):
        # return self.parameters_quantity
        total_parameter_quantity = self._parameter_quantity
        # for child in self.children:
        #     total_parameter_quantity += child.parameter_quantity
        return total_parameter_quantity

    @property
    def params_proportion(self):
        # total_parameter_quantity = 0
        # for child in self.parent.children:
        #     total_parameter_quantity += child._parameter_quantity
        # if hasattr(self.parent, 'root_total_params'):
        #     return (self._parameter_quantity / self.parent.root_total_params) * 100
        # elif hasattr(self.parent, 'p_total_params'):
        #     return (self._parameter_quantity / self.parent.p_total_params) * 100
        # else:
        #     return np.Inf

        try:
            return int((self._parameter_quantity / self.root_total_params)* 100), int((self._parameter_quantity / self.parent.p_total_params) * 100)
        except:
            return np.Inf

    @property
    def Flops_proportion(self):
        # total_Flops = 0
        # for child in self.parent.children:
        #     total_Flops += child.Flops

        # if hasattr(self.parent, 'root_total_flops'):
        #     return (self._Flops / self.parent.root_total_flops) * 100
        # elif hasattr(self.parent, 'p_total_flops'):
        #     return (self._Flops / self.parent.p_total_flops) * 100
        # else:
        #     return np.Inf

        try:
            return int((self._Flops / self.parent.root_total_flops) * 100), int((self._Flops / self.parent.p_total_flops) * 100)
        except:
            return np.Inf

    @parameter_quantity.setter
    def parameter_quantity(self, parameter_quantity):
        assert parameter_quantity >= 0
        self._parameter_quantity = parameter_quantity

    @property
    def inference_memory(self):
        total_inference_memory = self._inference_memory
        for child in self.children:
            total_inference_memory += child.inference_memory
        return total_inference_memory

    @inference_memory.setter
    def inference_memory(self, inference_memory):
        self._inference_memory = inference_memory

    @property
    def MAdd(self):
        total_MAdd = self._MAdd
        # for child in self.children:
        #     total_MAdd += child.MAdd
        return total_MAdd

    @MAdd.setter
    def MAdd(self, MAdd):
        self._MAdd = MAdd

    @property
    def Flops(self):
        total_Flops = self._Flops
        # for child in self.children:
        #     total_Flops += child.Flops
        return total_Flops

    @Flops.setter
    def Flops(self, Flops):
        self._Flops = Flops

    @property
    def Memory(self):
        total_Memory = self._Memory
        # for child in self.children:
        #     total_Memory[0] += child.Memory[0]
        #     total_Memory[1] += child.Memory[1]
        # print(total_Memory)
        return total_Memory

    @Memory.setter
    def Memory(self, Memory):
        assert isinstance(Memory, (list, tuple))
        self._Memory = Memory

    @property
    def duration(self):
        total_duration = self._duration
        # for child in self.children:
        #     total_duration += child.duration
        return total_duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    def find_child_index(self, child_name):
        assert isinstance(child_name, str)

        index = -1
        for i in range(len(self.children)):
            if child_name == self.children[i].name:
                index = i
        return index

    def add_child(self, node):
        assert isinstance(node, StatNode)

        if self.find_child_index(node.name) == -1:  # not exist
            self.children.append(node)
