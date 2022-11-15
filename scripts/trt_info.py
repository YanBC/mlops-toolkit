import sys
from typing import List, Tuple
from dataclasses import dataclass
import tensorrt as trt


@dataclass
class Binding:
    name: str
    dtype: trt.DataType
    shape: tuple
    size: int
    nbytes: int
    isInput: bool


@dataclass
class Profile:
    idx: int
    binding_name: str
    min_shape: Tuple[int]
    opt_shape: Tuple[int]
    max_shape: Tuple[int]


def get_binding_info(engine: trt.ICudaEngine) -> List[Binding]:
    num = engine.num_bindings
    max_batch_size = engine.max_batch_size
    bindings = []
    for idx in range(num):
        name = engine.get_binding_name(idx)
        dtype = engine.get_binding_dtype(idx)
        shape = engine.get_binding_shape(idx)
        size = abs(trt.volume(shape)) * max_batch_size
        nbytes = size * dtype.itemsize
        isInput = engine.binding_is_input(idx)
        binding = Binding(name, dtype, shape, size, nbytes, isInput)
        bindings.append(binding)
    return bindings


def load_engine(engine_file_path: str) -> trt.ICudaEngine:
    trt_logger = trt.Logger(trt.Logger.ERROR)
    # Force init TensorRT plugins
    trt.init_libnvinfer_plugins(None, '')
    with open(engine_file_path, "rb") as f, \
            trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def get_engine_info(engine: trt.ICudaEngine) -> str:
    name = engine.name
    max_batch_size = engine.max_batch_size
    num_bindings = engine.num_bindings
    bindings = get_binding_info(engine)
    engine_size = engine.device_memory_size
    input_host_mem_size = 0
    input_device_mem_size = 0
    output_host_mem_size = 0
    output_device_mem_size = 0
    for b in bindings:
        if b.isInput:
            input_device_mem_size += b.nbytes
            input_host_mem_size += b.nbytes
        else:
            output_device_mem_size += b.nbytes
            output_host_mem_size += b.nbytes
    host_mem_size = input_host_mem_size + output_host_mem_size
    device_mem_size = input_device_mem_size + output_device_mem_size

    num_optimization_profiles = engine.num_optimization_profiles
    profiles = []
    for profile_idx in range(num_optimization_profiles):
        profile = dict()
        for binding in bindings:
            if binding.isInput:
                binding_name = binding.name
                min_shape, opt_shape, max_shape = engine.get_profile_shape(profile_idx, binding_name)
                profile[binding_name] = Profile(
                    profile_idx,
                    binding_name,
                    min_shape,
                    opt_shape,
                    max_shape
                )
        profiles.append(profile)

    ret_str = ""
    ret_str += f"{name}\n"
    ret_str += f"Number of bindings: {num_bindings}\n"
    for idx, b in enumerate(bindings):
        ret_str += f"    #{idx} {b}\n"
    ret_str += f"Number of optimization profiles: {num_optimization_profiles}\n"
    for idx, p in enumerate(profiles):
        ret_str += f"    profile #{idx}\n"
        for input_name in p:
            ret_str += f"        {p[input_name]}\n"
    ret_str += f"Max batch size: {max_batch_size}\n"
    ret_str += f"Memory footprints:\n"
    ret_str += f"    Device memory: {device_mem_size/2**20:.2f} MiB\n"
    # ret_str += f"        Engine: {engine_size/2**20:.2f} MiB\n"
    ret_str += f"        Input:  {input_device_mem_size/2**20:.2f} MiB\n"
    ret_str += f"        Output: {output_device_mem_size/2**20:.2f} MiB\n"
    ret_str += f"    Host memory: {host_mem_size/2**20:.2f} MiB\n"
    ret_str += f"        Input:  {input_host_mem_size/2**20:.2f} MiB\n"
    ret_str += f"        Output: {output_host_mem_size/2**20:.2f} MiB\n"
    return ret_str


if __name__ == "__main__":
    engine_path = sys.argv[1]
    engine = load_engine(engine_path)
    info = get_engine_info(engine)
    print(info)
