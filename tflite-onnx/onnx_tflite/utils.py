import onnx
from onnx import helper


def tflite2onnx_shape_map(shape_list):
    # change dimension due to channel first-last issue
    if len(shape_list) != 4:
        return None
    return [shape_list[0], shape_list[3], shape_list[1], shape_list[2]]


def channel_last_2_channel_first_axis_mapping(axis_list):
    table = {'0': 0,
             '3': 1,
             '1': 2,
             '2': 3}
    res = []
    for axis in axis_list:
        res.append(table[str(axis)])
    return res


def get_output_node_info_by_name_if_exist(node_name, interpreter):
    output_details = interpreter.get_output_details()
    output_node_info_table = {}
    for node_info in output_details:
        if node_name == node_info['name']:
            return node_info
    return None


def getPadding(feat_map_size, kernel_size, strides, mode):
    # only support 'VALID' and 'SAME' mode
    if mode != 'VALID' and mode != 'SAME':
        return None

    # if mode is VALID
    if mode == 'VALID':
        return [0, 0, 0, 0]

    # else, mode is SAME 
    """ Calculate the padding array for same padding in the Tensorflow fashion.
    See https://www.tensorflow.org/api_guides/python/nn#Convolution for more.
    """
    if feat_map_size[1] % strides[0] == 0:
        pad_h = max(kernel_size[0] - strides[0], 0)
    else:
        pad_h = max(kernel_size[0] - feat_map_size[1] % strides[0], 0)

    if feat_map_size[2] % strides[1] == 0:
        pad_w = max(kernel_size[1] - strides[1], 0)
    else:
        pad_w = max(kernel_size[1] - feat_map_size[2] % strides[1], 0)

    return [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]


def make_kneron_valid_onnx_input(input_init):
    onnx_inputs = []
    for data in input_init:

        if isinstance(data, onnx.TensorProto):
            val = helper.make_tensor_value_info(
                data.name, data.data_type.real,
                list(d for d in data.dims))
            onnx_inputs.append(val)

        elif isinstance(data, onnx.AttributeProto):
            value_info = onnx.ValueInfoProto()
            value_info.name = data.name

            onnx_type = onnx_proto.TypeProto()
            onnx_type.tensor_type.elem_type = data.type
            value_info.type.CopyFrom(onnx_type)

            onnx_inputs.append(value_info)
        else:
            onnx_inputs.append(data)
    return onnx_inputs
