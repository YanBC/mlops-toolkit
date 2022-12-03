import argparse
from dataclasses import dataclass
from typing import Tuple
import onnx


def get_dtype_map() -> dict:
    str_names = onnx.TensorProto.DataType.keys()
    int_values = onnx.TensorProto.DataType.values()
    mapping = {
        v: n for v, n in zip(int_values, str_names)
    }
    return mapping


def parse_TensorShapeProto(shape: onnx.onnx_ml_pb2.TensorShapeProto) -> Tuple[int]:
    dims = []
    for d in shape.dim:
        v = d.dim_value
        if v == 0:
            dims.append(-1)
        else:
            dims.append(v)
    return tuple(dims)


@dataclass
class ModelIO:
    name: str
    dtype: str
    shape: Tuple[int]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('onnx_file', help="Path to onnx file")
    return p.parse_args()


def get_onnx_info(model: onnx.ModelProto) -> str:
    graph = model.graph

    dtype_map = get_dtype_map()

    opset_version = model.opset_import[0].version
    graph_name = graph.name
    model_inputs = []
    for input_io in graph.input:
        io_name = input_io.name
        io_dtype = dtype_map[input_io.type.tensor_type.elem_type]
        io_shape = parse_TensorShapeProto(input_io.type.tensor_type.shape)
        model_inputs.append(ModelIO(io_name, io_dtype, io_shape))
    model_outputs = []
    for output_io in graph.output:
        io_name = output_io.name
        io_dtype = dtype_map[output_io.type.tensor_type.elem_type]
        io_shape = parse_TensorShapeProto(output_io.type.tensor_type.shape)
        model_outputs.append(ModelIO(io_name, io_dtype, io_shape))

    ret_str = ''
    ret_str += f"Model Name: {graph_name}\n"
    ret_str += f"Opset version: {opset_version}\n"
    ret_str += f"Number of inputs: {len(model_inputs)}\n"
    for idx, input_io in enumerate(model_inputs):
        ret_str += f"    #{idx}: {input_io}\n"
    ret_str += f"Number of outputs {len(model_outputs)}\n"
    for idx, output_io in enumerate(model_outputs):
        ret_str += f"    #{idx}: {output_io}\n"

    return ret_str


if __name__ == "__main__":
    args = get_args()
    onnx_path = args.onnx_file
    model = onnx.load(onnx_path)
    print(get_onnx_info(model))
