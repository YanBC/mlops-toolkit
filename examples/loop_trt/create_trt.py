import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
cuda.init()

cuda_ctx = cuda.Device(0).make_context()
try:

    gLogger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(gLogger)
    trt_config = builder.create_builder_config()
    if builder.platform_has_fast_fp16:
        trt_config.set_flag(trt.BuilderFlag.FP16)
    trt_config.max_workspace_size = 1 << 23

    # build network
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    network.name = "Loop test"
    intput_tensor = network.add_input(name="x", dtype=trt.float32, shape=(1,))
    assert intput_tensor
    range_weight = trt.Weights(np.array([10], dtype=np.int32))
    constance = network.add_constant(shape=(), weights=range_weight)
    range_tensor = constance.get_output(0)
    assert range_tensor

    loop = network.add_loop()
    trip_limit = loop.add_trip_limit(
            tensor=range_tensor,
            kind=trt.TripLimit.COUNT
    )
    # index = loop.add_iterator(tensor=[i for i in range(10)])
    recurrence = loop.add_recurrence(intput_tensor)
    output_tensor = network.add_activation(input=recurrence.get_output(0), type=trt.ActivationType.SIGMOID).get_output(0)
    recurrence.set_input(1, output_tensor)

    loop_output = loop.add_loop_output(recurrence.get_output(0), kind=trt.LoopOutput.LAST_VALUE)

    network.mark_output(loop_output.get_output(0))

    # build engine
    engine = builder.build_engine(network, trt_config)

    context = engine.create_execution_context()
    device_in = cuda.mem_alloc(4)
    device_out = cuda.mem_alloc(4)
    host_in = cuda.pagelocked_empty(shape=(1,), dtype=np.float32)
    host_out = cuda.pagelocked_empty(shape=(1,), dtype=np.float32)

    np_input = np.array([1.2], dtype=np.float32)
    np_output = np_input.copy()
    for _ in range(10):
        np_output = 1 / (1 + np.exp(-1 * np_output))

    stream = cuda.Stream()
    np.copyto(np_input, host_in)
    cuda.memcpy_htod_async(device_in, host_in, stream)
    context.execute_async_v2(bindings=[device_in, device_out], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, device_out, stream)
    stream.synchronize()

    print(host_out)
    print(np_output)

finally:
    cuda_ctx.pop()
