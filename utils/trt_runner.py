from os.path import abspath
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import pycuda.driver as cuda
import tensorrt as trt
from utils.trt_profile import (
    Profile,
    query_profiles,
    get_cuda_mem_nbytes,
    load_engine
)


def allocate_cuda_mem_async(nbytes: int, stream: cuda.Stream) -> cuda.DeviceAllocation:
    cuda_mem = cuda.mem_alloc(nbytes)
    tmp = cuda.pagelocked_empty(1, np.uint8)
    cuda.memcpy_htod_async(cuda_mem, tmp, stream)
    return cuda_mem


def is_within_shape_range(shape: Tuple[int], min_shape: Tuple[int], max_shape: Tuple[int]) -> bool:
    assert len(shape) == len(min_shape) == len(max_shape)
    dim = len(shape)
    within = [min_shape[d] <= shape[d] <= max_shape[d] for d in range(dim)]
    return all(within)


def get_name_table(profiles: List[Profile]) -> dict:
    '''
    see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles_bindings
    '''
    ret = dict()
    default_profile = profiles[0]
    for binding in default_profile.inputs:
        ret[binding.name] = binding.idx
    for binding in default_profile.outputs:
        ret[binding.name] = binding.idx
    return ret


def reverse_table(table: dict) -> dict:
    ret = dict()
    for key in table.keys():
        ret[table[key]] = key
    return ret


def to_pagelocked(arr: np.ndarray) -> np.ndarray:
    '''
    This function has to be called after a valid cuda
    context is pushed into the stack
    '''
    pagelocked = cuda.pagelocked_empty(arr.shape, arr.dtype)
    np.copyto(pagelocked, arr)
    return pagelocked


@dataclass
class TRTInputLimit:
    min_shape: Tuple[int]
    max_shape: Tuple[int]

    def is_within(self, shape: Tuple[int]) -> bool:
        return is_within_shape_range(shape, self.min_shape, self.max_shape)

    def is_dynamic(self) -> bool:
        for idx in len(self.min_shape):
            if self.min_shape[idx] != self.max_shape[idx]:
                return True
        else:
            return False


# @dataclass
# class RunnerState:
#     engine_path: str
#     gpu_id: int
#     curr_profile: Profile


class TRTRunner:
    def __init__(self, engine_path: str, gpu_id: int, profile_idx: int = 0) -> None:
        profiles = query_profiles(engine_path=engine_path, gpu_id=gpu_id)

        cuda.init()
        cuda_ctx = cuda.Device(gpu_id).make_context()
        try:
            engine = load_engine(engine_path)
            exec_ctx = engine.create_execution_context()
            stream = cuda.Stream()
        finally:
            cuda_ctx.pop()

        self._engine_path = abspath(engine_path)
        self._gpu_id = gpu_id
        self._profile_idx = -1
        self._profiles = profiles
        self._cuda_ctx = cuda_ctx
        self._exec_ctx = exec_ctx
        self._stream = stream
        self._cuda_mem: List[cuda.DeviceAllocation] = []
        self.input_limits: List[TRTInputLimit] = []
        self.input_nptypes: List[np.dtype] = []
        self.set_profile(profile_idx)

    def set_profile(self, profile_idx: int) -> None:
        assert profile_idx < len(self._profiles), "invalid profile index"
        old_profile_idx = self._profile_idx
        if profile_idx == old_profile_idx:
            return
        profiles = self._profiles
        cuda_ctx = self._cuda_ctx
        exec_ctx = self._exec_ctx
        stream = self._stream
        old_cuda_mem = self._cuda_mem

        target_profile = profiles[profile_idx]
        p_inputs = target_profile.inputs
        p_outputs = target_profile.outputs
        new_input_limit_list = [TRTInputLimit(b.min_shape, b.max_shape) for b in p_inputs]
        new_input_np_dtype_list = [trt.nptype(b.dtype) for b in p_inputs]

        io_nbytes_list = [get_cuda_mem_nbytes(b) for b in p_inputs+p_outputs]
        new_cuda_mem = []

        cuda_ctx.push()
        try:
            # set optimization profile
            exec_ctx.set_optimization_profile_async(profile_idx, stream.handle)

            # allocate new cuda memories
            for nbytes in io_nbytes_list:
                cuda_mem = allocate_cuda_mem_async(nbytes, stream)
                new_cuda_mem.append(cuda_mem)

            # stream synchronize
            stream.synchronize()

            # update self
            self._profile_idx = profile_idx
            self._cuda_mem = new_cuda_mem
            self.input_limits = new_input_limit_list
            self.input_nptypes = new_input_np_dtype_list

            # free old cuda memories
            while len(old_cuda_mem) > 0:
                mem_alloc = old_cuda_mem.pop()
                mem_alloc.free()
                del mem_alloc

        finally:
            cuda_ctx.pop()

    def inference(self, np_inputs: List[np.ndarray]) -> List[np.ndarray]:
        input_limits: List[TRTInputLimit] = self.input_limits
        input_nptypes: List[np.ndarray] = self.input_nptypes
        cuda_ctx = self._cuda_ctx
        exec_ctx = self._exec_ctx
        cuda_mem = self._cuda_mem
        stream = self._stream
        profile = self._profiles[self._profile_idx]

        for idx in range(len(np_inputs)):
            assert input_limits[idx].is_within(np_inputs[idx].shape)

        cuda_ctx.push()
        try:
            host_mem_inputs = []
            for idx in range(len(np_inputs)):
                np_inputs[idx] = np_inputs[idx].astype(input_nptypes[idx])
                np_inputs[idx] = np.ascontiguousarray(np_inputs[idx])
                host_mem_inputs.append(to_pagelocked(np_inputs[idx]))

            for idx in range(len(host_mem_inputs)):
                input_binding = profile.inputs[idx]
                host_mem_input = host_mem_inputs[idx]
                exec_ctx.set_binding_shape(input_binding.idx, host_mem_input.shape)

            host_mem_outputs = []
            for idx in range(len(profile.outputs)):
                output_binding = profile.outputs[idx]
                shape = tuple(exec_ctx.get_binding_shape(output_binding.idx))
                dtype = output_binding.np_dtype
                host_mem_outputs.append(cuda.pagelocked_empty(shape, dtype))

            for input_idx in range(len(host_mem_inputs)):
                cuda.memcpy_htod_async(
                        cuda_mem[input_idx],
                        host_mem_inputs[input_idx],
                        stream,
                )
            exec_ctx.execute_async_v2(
                    bindings=cuda_mem,
                    stream_handle=stream.handle,
            )
            for output_idx in range(len(host_mem_outputs)):
                cuda.memcpy_dtoh_async(
                        host_mem_outputs[output_idx],
                        cuda_mem[input_idx+1+output_idx],
                        stream,
                )
            stream.synchronize()

            return host_mem_outputs
        finally:
            cuda_ctx.pop()

    # def get_state(self) -> str:
    #     pass
