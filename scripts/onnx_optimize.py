import sys
import argparse
import onnx
import onnx_graphsurgeon as gs


############
# funcs
############
def load_ModelProto(path: str) -> onnx.ModelProto:
    model = onnx.load(path)
    return model


def save_ModelProto(path: str, model: onnx.ModelProto) -> None:
    onnx.save(model, path)


def prune_onnx(src_path, des_path):
    graph = gs.import_onnx(onnx.load(src_path))
    graph.cleanup().toposort().fold_constants().cleanup()
    onnx.save_model(gs.export_onnx(graph), des_path)


def shape_ModelProto(model: onnx.ModelProto):
    new_model = onnx.shape_inference.infer_shapes(model)
    return model


def check_ModelProto(model: onnx.ModelProto):
    onnx.checker.check_model(model)


############
# main
############
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("onnx_file", help="path to onnx model file")
    return p.parse_args()

if __name__ == '__main__':
    args = get_args()
    onnx_path = args.onnx_file

    # onnx_name = onnx_path.split("/")[-1]
    # prune_onnx(onnx_path, onnx_name)

    model = load_ModelProto(onnx_path)
    check_ModelProto(model)
    model = shape_ModelProto(model)
    save_ModelProto("res.onnx", model)
