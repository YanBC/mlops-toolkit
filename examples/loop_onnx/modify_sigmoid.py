import onnx


##############################
# helper functions
##############################
def save_onnx_graph(filepath: str, graph: onnx.GraphProto) -> None:
    model = onnx.helper.make_model(
            graph,
            producer_name="yanbc"
    )
    model.opset_import[0].version = 13
    model = onnx.shape_inference.infer_shapes(model)

    onnx.checker.check_model(model)
    onnx.save(model, filepath)


##############################
# load model
##############################
sigmoid_model = onnx.load("sigmoid.onnx")
onnx.checker.check_model(sigmoid_model)

##############################
# make loop body
##############################
## valueinfos
x = sigmoid_model.graph.input[0]
y = sigmoid_model.graph.output[0]
count = onnx.helper.make_tensor_value_info(
    "body.count", onnx.TensorProto.INT64, ()
)
cond_in = onnx.helper.make_tensor_value_info(
    "body.cond_in", onnx.TensorProto.BOOL, ()
)
cond_out = onnx.helper.make_tensor_value_info(
    "body.cond_out", onnx.TensorProto.BOOL, ()
)
factor_in = onnx.helper.make_tensor_value_info(
    "body.factor_in", onnx.TensorProto.FLOAT, ()
)
factor_out = onnx.helper.make_tensor_value_info(
    "body.factor_out", onnx.TensorProto.FLOAT, ()
)
result = onnx.helper.make_tensor_value_info(
    "body.result", onnx.TensorProto.FLOAT, (None, 1)
)
## nodes and initializers
initializers = sigmoid_model.graph.initializer
nodes = sigmoid_model.graph.node
cond_identity_node = onnx.helper.make_node(
        "Identity",
        inputs=["body.cond_in"],
        outputs=["body.cond_out"],
)
nodes.append(cond_identity_node)
mul_node = onnx.helper.make_node(
        "Mul",
        inputs=["body.factor_in", y.name],
        outputs=["body.result"]
)
nodes.append(mul_node)
factor_identity_node = onnx.helper.make_node(
        "Identity",
        inputs=["body.factor_in"],
        outputs=["body.factor_out"],
)
nodes.append(factor_identity_node)
## graph
## for a graph to be used as the body of a
## loop, it has to have 2+N inputs and 1+N
## outputs
loop_body_graph = onnx.helper.make_graph(
        nodes=nodes,
        name="sigmoid loop body",
        inputs=[count, cond_in, x, factor_in],
        outputs=[cond_out, result, factor_out],
        initializer=initializers,
)
## save onnx
save_onnx_graph("loop_body.onnx", loop_body_graph)

##############################
# make loop node and graph
##############################
# loop_count = onnx.helper.make_tensor_value_info(
#     "loop_count", onnx.TensorProto.INT64, ()
# )
loop_count = onnx.helper.make_tensor(
    "loop.count", onnx.TensorProto.INT64, (),
    vals=[10]
)
# loop_cond = onnx.helper.make_tensor_value_info(
#     "loop_cond", onnx.TensorProto.BOOL, ()
# )
loop_cond = onnx.helper.make_tensor(
    "loop.cond", onnx.TensorProto.BOOL, (),
    vals=[True]
)
loop_input = onnx.helper.make_tensor_value_info(
    "loop.input", onnx.TensorProto.FLOAT, (None, 1)
)
loop_output = onnx.helper.make_tensor_value_info(
    "loop.output", onnx.TensorProto.FLOAT, (None, 1)
)
loop_factor_in = onnx.helper.make_tensor_value_info(
    "loop.factor", onnx.TensorProto.FLOAT, ()
)
_ = onnx.helper.make_tensor_value_info(
    "loop.factor_out", onnx.TensorProto.FLOAT, ()
)
loop = onnx.helper.make_node(
        op_type="Loop",
        inputs=["loop.count", "loop.cond", "loop.input", "loop.factor"],
        outputs=["loop.output", "loop.factor_out"],
        body=loop_body_graph
)
graph = onnx.helper.make_graph(
        nodes=[loop],
        name="loop sigmoid",
        inputs=[loop_input, loop_factor_in],
        outputs=[loop_output],
        initializer=[loop_count, loop_cond]
)
## save onnx
save_onnx_graph("loop_sigmoid.onnx", graph)
