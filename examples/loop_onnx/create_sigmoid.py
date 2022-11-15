import onnx


if __name__ == "__main__":
    # ValueInfoProto
    model_input_name = "X"
    X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, 1]
    )

    model_output_name = "Y"
    Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 1]
    )

    # NodeProto
    sigmoid_node = onnx.helper.make_node(
            name="sigmoid",
            op_type="Sigmoid",
            inputs=[model_input_name],
            outputs=[model_output_name]
    )

    # GraphProto
    graph = onnx.helper.make_graph(
            nodes=[sigmoid_node],
            name="Demo",
            inputs=[X],
            outputs=[Y],
            initializer=[]
    )

    # ModelProto
    model = onnx.helper.make_model(
            graph,
            producer_name="yanbc"
    )
    model.opset_import[0].version = 13
    model = onnx.shape_inference.infer_shapes(model)

    # check and save
    onnx.checker.check_model(model)
    onnx.save(model, "sigmoid.onnx")
