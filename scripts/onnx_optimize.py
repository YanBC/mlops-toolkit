import sys
import onnx
import onnx_graphsurgeon as gs


def prune_onnx(src_path, des_path):
    graph = gs.import_onnx(onnx.load(src_path))
    graph.cleanup().toposort().fold_constants().cleanup()
    onnx.save_model(gs.export_onnx(graph), des_path)


if __name__ == '__main__':
    src_path = sys.argv[1]
    onnx_name = src_path.split("/")[-1]
    prune_onnx(src_path, onnx_name)
