import numpy as np
from utils.trt_runner import TRTRunner


##############
# meta
##############
engine_path = "/yanbc/codes/mlops-toolkit/examples/trt_multi_profiles/res.engine"
gpu_id = 0
profile_idx = 0
np_inputs = [
    np.random.rand(1, 3, 32, 32).astype(np.float32)
]


##############
# main
##############
if __name__ == "__main__":
    runner = TRTRunner(
        engine_path=engine_path,
        gpu_id=gpu_id,
        profile_idx=profile_idx
    )
    np_outputs = runner.inference(np_inputs)

    # print(np_outputs)
