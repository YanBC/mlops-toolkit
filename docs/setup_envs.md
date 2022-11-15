# Create containers
## pytorch
```bash
docker run --init -it -d \
    --name yanbc-pytorch-21.07 \
    --gpus all \
    -v /data1/yanbc:/yanbc \
    --net host \
    --ipc host \
    nvcr.io/nvidia/pytorch:21.07-py3 \
    bash
```

## triton server
```bash
docker run --init -it -d \
    --name yanbc-triton-server-21.07 \
    --gpus all \
    -v /data1/yanbc:/yanbc \
    --net host \
    --ipc host \
    nvcr.io/nvidia/tritonserver:21.07-py3 \
    bash
```

## triton client
```bash
docker run --init -it -d \
    --name yanbc-triton-client-21.07 \
    --gpus all \
    -v /data1/yanbc:/yanbc \
    --net host \
    --ipc host \
    nvcr.io/nvidia/tritonserver:21.07-py3-sdk \
    bash
```

```bash
pip install --upgrade pip && \
pip install onnx-graphsurgeon && \
pip install onnxruntime && \
pip install pycuda
```

# Run commands
## triton server
```bash
export CUDA_VISIBLE_DEVICES=3
tritonserver --http-port 18000 \
             --grpc-port 18001 \
             --model-repository model-repository
```
