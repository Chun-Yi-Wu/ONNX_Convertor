"""Base layer class
"""
import abc
import logging
import json


class Layer(metaclass=abc.ABCMeta):

    def __init__(self, op, op_type, tflite_interpreter):
        # basic parse
        self.op = op
        self.op_type = op_type
        self.tflite_interpreter = tflite_interpreter

        self.__init_node_info()
        self.__init_inputs()
        self.__init_outputs()

    def __init_node_info(self):
        # Only None for Generated Fused Node
        if self.op is None:
            self.node_idx = -1
            self.node_name = 'fused_node'
            return

        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))

        self.node_idx = node_output_detail['index']
        self.node_name = node_output_detail['name']

    def __init_inputs(self):
        # Only None for Generated Fused Node
        if self.op is None:
            self.input_nodes_idx = []
            self.input_nodes_name = []
            return

        self.input_nodes_idx = [self.op.Inputs(i) for i in range(self.op.InputsLength())]
        self.input_nodes_name = [self.tflite_interpreter._get_tensor_details(idx)['name'] for idx in self.input_nodes_idx]

    def __init_outputs(self):
        self.output_nodes_idx = []
        self.output_nodes_name = []

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        # basic parse
        self.op_type = op_type
        self.op_info = op_info
        self.tflite_interpreter = tflite_interpreter

        self.logger = logging.getLogger("onnx-tflite")
        self.logger.setLevel(logging.DEBUG)

        self.node_output_detail = tflite_interpreter._get_tensor_details(self.op_info.Outputs(0))
        self.node_input_detail = tflite_interpreter._get_tensor_details(self.op_info.Inputs(0))

        self.onnx_node_name = self.node_output_detail['name']
        self.previous_onnx_node_names = previous_onnx_node_names

        self.node_list = []
        self.value_infos = []
        self.weight_node_list = []
        return self

    @abc.abstractmethod
    def generate(self):
        """Generate the nodes according to the original layer

         Returns
        -------
        node_list : list
            a list of nodes generated
        value_infos : list
            value_infos between nodes
        """
        return NotImplemented

    def defuse_activation_function(self):
        '''
        de-fuse activation node from current node

        :return:
            layer: Layer
                de-fused activation node, if not contain fused activation will return None.
        '''
        return None

    def __repr__(self):
        show_dict = {
            'idx': self.node_idx,
            'name': self.node_name,
            'input_nodes_idx': self.input_nodes_idx,
            'input_nodes_name': self.input_nodes_name,
            'output_nodes_idx': self.output_nodes_idx,
            'output_nodes_name': self.output_nodes_name,
        }
        return json.dumps(show_dict, indent=4, sort_keys=True)

    def __str__(self):
        show_dict = {
            'idx': self.node_idx,
            'name': self.node_name,
            'input_nodes_idx': self.input_nodes_idx,
            'input_nodes_name': self.input_nodes_name,
            'output_nodes_idx': self.output_nodes_idx,
            'output_nodes_name': self.output_nodes_name,
        }
        return json.dumps(show_dict, indent=4, sort_keys=True)
