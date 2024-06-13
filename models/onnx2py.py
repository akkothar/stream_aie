import onnx
import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = "/home/akkothar/stream_aie/models/matmul_288_1_288.onnx"
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)
