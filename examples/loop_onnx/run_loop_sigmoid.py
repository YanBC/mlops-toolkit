import onnxruntime
import numpy as np


onnx_model_path = "loop_sigmoid.onnx"
so = onnxruntime.SessionOptions()
so.log_severity_level = 3

with open(onnx_model_path, 'rb') as f:
    model_bytes = f.read()
sess = onnxruntime.InferenceSession(model_bytes, so)
# sess = onnxruntime.InferenceSession(onnx_model_path, so)

# model specific
np_input = np.array([1.2], dtype=np.float32).reshape((1, 1))
# np_factor = np.array([2], dtype=np.float32).reshape((1, 1))
np_factor = np.array([2], dtype=np.float32)
inputs = {
    "loop.input": np_input,
    "loop.factor": np_factor
}
sess_out = sess.run(None, inputs)
print(sess_out)

np_output = np_input.copy()
for _ in range(10):
    np_output = 1 / (1 + np.exp(-1 * np_output))
    np_output *= np_factor
print(np_output)
