import os
import argparse
from typing import Tuple, List
import tensorrt as trt
import onnx


##############
# meta
##############
_WORKSPACE = 8<<30
_USE_FP16 = False
_TRT_LOG_LEVEL = trt.Logger.VERBOSE
# !!! require manual setting for now
# _PROFILES = [
#     {
#         "X": [(1, 3, 32, 32), (1, 3, 32, 32), (1, 3, 32, 32)],              # static
#     },
#     {
#         "X": [(1, 3, 64, 64), (2, 3, 64, 64), (4, 3, 64, 64)],              # dynamic batch size
#     },
#     {
#         "X": [(1, 3, 32, 32), (1, 3, 64, 64), (1, 3, 128, 128)],            # dynamic height and width
#     },
#     {
#         "X": [(1, 3, 32, 32), (2, 3, 64, 64), (4, 3, 128, 128)],            # dynamic batch size, height, and width
#     }
# ]
_PROFILES = [
    {
        "input_ids": [(1, 1), (1, 16), (1, 128)],
    }
]


##############
# helpers
##############
def build_engine(onnx_file_path,
                engine_file_path,
                engine_name,
                optimization_profiles: List[dict],
                use_fp16=True,
                workspace_size=_WORKSPACE
    ):
    trt_logger = trt.Logger(_TRT_LOG_LEVEL)
    builder = trt.Builder(trt_logger)

    # create network
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
    if len(optimization_profiles) > 0:
        for op in optimization_profiles:
            builder_profile = builder.create_optimization_profile()
            for binding_name, shapes in op.items():
                min_shape, opt_shape, max_shape = shapes
                builder_profile.set_shape(
                    binding_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(builder_profile)

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
    p.add_argument("--enginepath", default="res.engine", help="Where to save tensorrt engine file. Default to res.engine")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    onnx_path = args.onnxpath
    engine_path = args.enginepath
    profiles = _PROFILES

    # parse onnx file
    dtype_map = get_onnx_dtype_map()
    onnx_graph = onnx.load(onnx_path).graph
    model_name = onnx_graph.name

    build_engine(
        onnx_path,
        engine_path,
        model_name,
        optimization_profiles=profiles,
        use_fp16=_USE_FP16,
    )
