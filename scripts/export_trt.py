import os
import argparse
from typing import Tuple
import tensorrt as trt
import onnx


##############
# meta
##############
_WORKSPACE = 8<<30


##############
# helpers
##############
def build_engine(onnx_file_path,
                engine_file_path,
                engine_name,
                use_fp16=True,
                dynamic_shapes={},
                workspace_size=_WORKSPACE
    ):
    """Build TensorRT Engine

    :use_fp16: set mixed flop computation if the platform has fp16.
    :dynamic_shapes: {binding_name: (min, opt, max)}, default {} represents not using dynamic.
    :dynamic_batch_size: set it to 1 if use fixed batch size, else using max batch size
    """
    use_dynamic_shapes = len(dynamic_shapes) > 0
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)

    # create network
    # if use_dynamic_shapes:
    #     network = builder.create_network(
    #         1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # else:
    #     network = builder.create_network()
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network.name = engine_name

    # create config
    config = builder.create_builder_config()
    config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)
    config.max_workspace_size = workspace_size
    if builder.platform_has_fast_fp16 and use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # set tensor shapes
    # builder.max_batch_size = batch_size
    if use_dynamic_shapes:
        profile = builder.create_optimization_profile()
        for binding_name, dynamic_shape in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = dynamic_shape
            profile.set_shape(
                binding_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # parse ONNX
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("===> Completed parsing ONNX file")

    # create engine
    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print(f"Cannot remove existing file: {engine_file_path}")
    print("===> Creating Tensorrt Engine...")
    engine = builder.build_engine(network, config)
    if engine:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("===> Serialized Engine Saved at: ", engine_file_path)
    else:
        print("===> build engine error")

    return engine


def get_onnx_dtype_map() -> dict:
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


##############
# main
##############
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("onnxpath", help="Path to onnx model file")
    p.add_argument("--batchsize", type=int, default=1, help="Batch size. Defualt to 1.")
    p.add_argument("--enginepath", default="res.engine", help="Where to save tensorrt engine file. Default to res.engine")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    onnx_path = args.onnxpath
    batch_size = args.batchsize
    engine_path = args.enginepath

    # parse onnx file
    dtype_map = get_onnx_dtype_map()
    onnx_graph = onnx.load(onnx_path).graph
    model_name = onnx_graph.name
    dynamic_shapes = dict()
    if batch_size != 1:
        for input_io in onnx_graph.input:
            io_name = input_io.name
            io_dtype = dtype_map[input_io.type.tensor_type.elem_type]
            io_shape = parse_TensorShapeProto(input_io.type.tensor_type.shape)
            io_shape = io_shape[1:]     # remove leading batch
            dynamic_shapes[io_name] = [
                (1, *io_shape), (batch_size, *io_shape), (batch_size, *io_shape)
            ]

    build_engine(
        onnx_path,
        engine_path,
        model_name,
        dynamic_shapes=dynamic_shapes,
        use_fp16=True
    )
