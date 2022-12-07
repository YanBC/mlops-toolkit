import argparse
import onnx
import onnxruntime
import onnx_graphsurgeon as gs
import numpy as np


################
# meta
################
inputs = {
    "input": np.random.rand(1, 3, 512, 512).astype(np.float32)
}


################
# funcs
################
def load_ModelProto(path: str) -> onnx.ModelProto:
    model = onnx.load(path)
    return model


def save_ModelProto(path: str, model: onnx.ModelProto) -> None:
    try:
        onnx.save(model, path)
    except ValueError:
        # ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB: 3773054338
        onnx.save(model, path,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            convert_attribute=True,
        )


def prune_ModelProto(model: onnx.ModelProto) -> onnx.ModelProto:
    gs_graph = gs.import_onnx(model)
    gs_graph.cleanup().toposort().fold_constants().cleanup()
    return gs.export_onnx(gs_graph)


def create_model(name, nodes, inputs, outputs, initializers, opset_num) -> onnx.ModelProto:
    new_graph = onnx.helper.make_graph(
            nodes=nodes,
            name=name,
            inputs=inputs,
            outputs=outputs,
            initializer=initializers
    )
    new_model = onnx.helper.make_model(new_graph)
    new_model.opset_import[0].version = opset_num

    new_model = prune_ModelProto(new_model)
    try:
        new_model = onnx.shape_inference.infer_shapes(new_model)
        onnx.checker.check_model(new_model)
    except ValueError as ve:
        print(ve)
    return new_model


################
# main
################
def modify_output(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    new_output = onnx.helper.make_tensor_value_info(
            "new_output",
            onnx.TensorProto.FLOAT,
            [4]
    )
    copy_node = onnx.helper.make_node(
            name="copy",
            op_type="Identity",
            inputs=["onnx::Resize_1938"],
            outputs=["new_output"]
    )

    nodes = graph.node
    nodes.append(copy_node)
    initializers = graph.initializer
    inputVIs = graph.input
    outputVIs = [new_output]
    opset = model.opset_import[0].version

    new_model = create_model("Modified", nodes, inputVIs, outputVIs, initializers, opset)
    return new_model


def modify_input(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    opset = model.opset_import[0].version
    name = graph.name
    inputVIs = graph.input
    outputVIs = graph.output
    nodes = graph.node
    initializers = graph.initializer

    extra_dim = onnx.TensorShapeProto.Dimension()
    extra_dim.dim_value = 1

    vi_t = inputVIs[2]
    vi_index = inputVIs[3]
    vi_scale = inputVIs[4]

    vi_t.type.tensor_type.shape.dim.append(extra_dim)
    vi_index.type.tensor_type.shape.dim.append(extra_dim)
    vi_scale.type.tensor_type.shape.dim.append(extra_dim)

    new_model = create_model(name, nodes, inputVIs, outputVIs, initializers, opset)
    return new_model


def modify_node(model: onnx.ModelProto) -> onnx.ModelProto:
    '''
    replace any node input "onnx::Resize_1938" -> "onnx::Resize_1937"
    '''
    graph = model.graph

    name = graph.name
    nodes = graph.node
    initializers = graph.initializer
    inputVIs = graph.input
    outputVIs = graph.output
    opset = model.opset_import[0].version

    target_tensor_name = "onnx::Resize_1938"
    target_node = [n for n in nodes if target_tensor_name in n.input][0]
    target_node.input.pop()
    target_node.input.append("onnx::Resize_1937")

    new_model = create_model(name, nodes, inputVIs, outputVIs, initializers, opset)
    return new_model


def modify_opset(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph

    name = graph.name
    nodes = graph.node
    initializers = graph.initializer
    inputVIs = graph.input
    outputVIs = graph.output
    opset = 13

    new_model = create_model(name, nodes, inputVIs, outputVIs, initializers, opset)
    return new_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('onnx_file', help="Path to onnx file")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    onnx_path = args.onnx_file
    model = load_ModelProto(onnx_path)
    # model = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)

    # new_model = modify_output(model)
    # new_model = modify_node(model)
    new_model = modify_opset(model)
    # new_model = modify_input(model)

    save_ModelProto("res.onnx", new_model)

    # # run
    # options = onnxruntime.SessionOptions()
    # options.log_severity_level = 3
    # sess = onnxruntime.InferenceSession(model.SerializeToString(), options)

    # sess_out = sess.run(["new_output"], inputs)
