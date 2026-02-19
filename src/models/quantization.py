import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig

def build_quant_config(quantization: str):
    if quantization == "none":
        return None, {}
    
async def torch_quantize(model: nn.Module, quantization: str):
    return quantize_(model, Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8))