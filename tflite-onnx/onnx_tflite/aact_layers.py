"""Converters for core layers in TFlite
"""
import abc
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto
from tflite.ActivationFunctionType import ActivationFunctionType
import numpy as np
from base_layer import Layer
import utils
import warnings


def defused_activation_node_generator(activation_function_type: int, op, op_type, tflite_interpreter):
    if ActivationFunctionType.NONE == activation_function_type:
        return None
    elif ActivationFunctionType.RELU == activation_function_type:
        return ReluDefused(op=op, op_type=op_type, tflite_interpreter=tflite_interpreter)
    elif ActivationFunctionType.RELU_N1_TO_1 == activation_function_type:
        warnings.warn('Not Support {} Fused Activation Currently.'.format('RELU_N1_TO_1'),
                      UserWarning)
        return None
    elif ActivationFunctionType.RELU6 == activation_function_type:
        return Relu6Defused(op=op, op_type=op_type, tflite_interpreter=tflite_interpreter)
    elif ActivationFunctionType.TANH == activation_function_type:
        warnings.warn('Not Support {} Fused Activation Currently.'.format('TANH'),
                      UserWarning)
        return None
    elif ActivationFunctionType.SIGN_BIT == activation_function_type:
        warnings.warn('Not Support {} Fused Activation Currently.'.format('SIGN_BIT'),
                      UserWarning)
        return None
    else:
        warnings.warn('Fused Activation Type {} Not in Specification.'.format(activation_function_type),
                      UserWarning)
        return None


# Defused Activation Layer
class ActivationDefused(Layer, metaclass=abc.ABCMeta):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)
        self.node_name = '{}_Fused'.format(self.node_name)

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        return Layer.init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)


class ReluDefused(ActivationDefused):

    def __init__(self, op, op_type, tflite_interpreter):
        ActivationDefused.__init__(self, op, op_type, tflite_interpreter)

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        return ActivationDefused.init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        relu_name = self.node_name
        relu_node = helper.make_node("Relu",
                                     name=relu_name,
                                     inputs=[self.onnx_node_name],
                                     outputs=[relu_name])

        out_shape_info = onnx.helper.make_tensor_value_info(
            relu_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(relu_node)

        return self.node_list, self.value_infos, self.weight_node_list


class Relu6Defused(ActivationDefused):

    def __init__(self, op, op_type, tflite_interpreter):
        ActivationDefused.__init__(self, op, op_type, tflite_interpreter)

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        return ActivationDefused.init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        clip_name = self.node_name
        clip_node = helper.make_node(
            'Clip',
            inputs=[self.onnx_node_name],
            outputs=[clip_name],
            min=0.0,
            max=6.0,
            name=clip_name)

        out_shape_info = helper.make_tensor_value_info(
            clip_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(clip_node)

        return self.node_list, self.value_infos, self.weight_node_list


# Normal Activation Layer
class Relu(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        return Layer.init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        relu_node = helper.make_node(
            "Relu",
            name=self.onnx_node_name,
            inputs=self.previous_onnx_node_names,
            outputs=[self.onnx_node_name]
        )

        # original layer output
        out_shape_info = onnx.helper.make_tensor_value_info(
            self.onnx_node_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(relu_node)

        return self.node_list, self.value_infos, self.weight_node_list


class Relu6(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        return Layer.init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        clip_node = onnx.helper.make_node(
            'Clip',
            inputs=self.previous_onnx_node_names,
            outputs=[self.onnx_node_name],
            min=0.0,
            max=6.0,
            name=self.onnx_node_name
        )

        # original layer output
        out_shape_info = onnx.helper.make_tensor_value_info(
            self.onnx_node_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(clip_node)

        return self.node_list, self.value_infos, self.weight_node_list


class LOGISTIC(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        return Layer.init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        logistic_name = self.onnx_node_name
        logistic_node = helper.make_node(
            op_type='Sigmoid',
            inputs=self.previous_onnx_node_names,
            outputs=[logistic_name],
            name=logistic_name
        )
        self.node_list.append(logistic_node)

        return self.node_list, self.value_infos, self.weight_node_list


class Softmax(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        return Layer.init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        softmax_node_name = self.onnx_node_name
        softmax_node = onnx.helper.make_node(
            'Softmax',
            inputs=self.previous_onnx_node_names,
            outputs=[softmax_node_name],
            name=softmax_node_name
        )
        self.node_list.append(softmax_node)

        return self.node_list, self.value_infos, self.weight_node_list


class PRelu(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        return Layer.init_generate(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        slope_node_info = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(1))
        slope_array = self.tflite_interpreter.get_tensor(slope_node_info['index'])
        slope_array = np.transpose(slope_array, (2, 0, 1))

        # make slope onnx node
        slope_onnx_node_name = self.onnx_node_name + "_slope"
        slope_onnx_node = onnx.helper.make_tensor(
            slope_onnx_node_name,
            TensorProto.FLOAT,
            slope_array.shape,
            slope_array.flatten().tolist()
        )
        self.weight_node_list.append(slope_onnx_node)

        self.previous_onnx_node_names.extend([slope_onnx_node_name])
        prelu_node = onnx.helper.make_node(
            'PRelu',
            inputs=self.previous_onnx_node_names,
            outputs=[self.onnx_node_name],
            name=self.onnx_node_name
        )
        self.node_list.append(prelu_node)

        # original layer output
        out_shape_info = onnx.helper.make_tensor_value_info(
            self.onnx_node_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
        )
        self.value_infos.append(out_shape_info)

        return self.node_list, self.value_infos, self.weight_node_list
