from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
import tensorrt as trt
import pycuda.driver as cuda


def load_engine(engine_file_path: str) -> trt.ICudaEngine:
    trt_logger = trt.Logger(trt.Logger.ERROR)
    # Force init TensorRT plugins
    trt.init_libnvinfer_plugins(None, '')
    with open(engine_file_path, "rb") as f, \
            trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def save_engine(engine_file_path: str, engine: trt.ICudaEngine):
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())


@dataclass
class Binding:
    idx: int                    # engine.get_binding_shape
    name: str                   # engine.get_binding_shape
    dtype: trt.DataType
    min_shape: Tuple[int]
    opt_shape: Tuple[int]
    max_shape: Tuple[int]       # output binding only has max_shape
    isInput: bool
    isDynamic: bool

    @property
    def np_dtype(self) -> np.dtype:
        return trt.nptype(self.dtype)


def get_nbytes(shape: Tuple[int], dtype: trt.DataType) -> int:
    size = abs(trt.volume(shape))
    nbytes = size * dtype.itemsize
    return nbytes


def get_cuda_mem_nbytes(binding: Binding) -> int:
    max_nbytes = get_nbytes(binding.max_shape, binding.dtype)
    return max_nbytes


def is_shape_dynamic(shape: Tuple[int]) -> bool:
    dynamic = [d == -1 for d in shape]
    return any(dynamic)


@dataclass
class Profile:
    idx: int
    inputs: List[Binding]
    outputs: List[Binding]

    def is_dynamic(self) -> bool:
        input_dynamic = [b.isDynamic for b in self.inputs]
        return any(input_dynamic)


def infer_output_shape(profile: Profile, engine: trt.ICudaEngine):
    profile_idx = profile.idx
    context = engine.create_execution_context()
    stream = cuda.Stream()
    try:
        context.set_optimization_profile_async(profile_idx, stream.handle)
        stream.synchronize()

        # min_shape
        for binding in profile.inputs:
            context.set_binding_shape(binding.idx, binding.min_shape)

        for binding in profile.outputs:
            binding.min_shape = context.get_binding_shape(binding.idx)

        # opt_shape
        for binding in profile.inputs:
            context.set_binding_shape(binding.idx, binding.opt_shape)

        for binding in profile.outputs:
            binding.opt_shape = context.get_binding_shape(binding.idx)

        # max_shape
        for binding in profile.inputs:
            context.set_binding_shape(binding.idx, binding.max_shape)

        for binding in profile.outputs:
            binding.max_shape = context.get_binding_shape(binding.idx)

    finally:
        del stream
        del context


def get_profiles(engine: trt.ICudaEngine) -> List[Profile]:
    num_bindings = engine.num_bindings
    num_profiles = engine.num_optimization_profiles
    num_bindings_per_profile = num_bindings // num_profiles

    ret_profiles = []
    count = 0
    profile_idx = 0
    input_bindings = []
    output_bindings = []
    for binding_idx in range(num_bindings):
        name = engine.get_binding_name(binding_idx)
        dtype = engine.get_binding_dtype(binding_idx)
        shape: trt.Dims = engine.get_binding_shape(binding_idx)
        shape: Tuple[int] = tuple(shape)
        if engine.binding_is_input(binding_idx):
            min_shape, opt_shape, max_shape = engine.get_profile_shape(profile_idx, binding_idx)
            isInput = True
        else:
            min_shape = [-1]
            opt_shape = [-1]
            max_shape = [-1]
            isInput = False

        binding = Binding(
                idx=binding_idx,
                name=name,
                dtype=dtype,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                isInput=isInput,
                isDynamic=is_shape_dynamic(shape)
        )

        if isInput:
            input_bindings.append(binding)
        else:
            output_bindings.append(binding)

        count += 1
        if count == num_bindings_per_profile:
            profile = Profile(
                idx=profile_idx,
                inputs=input_bindings,
                outputs=output_bindings,
            )
            ret_profiles.append(profile)

            count = 0
            profile_idx += 1
            input_bindings = []
            output_bindings = []

    for profile in ret_profiles:
        infer_output_shape(profile, engine)
    return ret_profiles


#################
# main
#################
def query_profiles(engine_path: str, gpu_id: int = 0) -> List[Profile]:
    cuda.init()
    cuda_ctx = cuda.Device(gpu_id).make_context()
    engine = None
    try:
        engine = load_engine(engine_path)
        profiles = get_profiles(engine)
        return profiles
    finally:
        if engine is not None:
            del engine
        cuda_ctx.pop()
