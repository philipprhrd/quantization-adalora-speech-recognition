import torch

def build_quant_config(quantization: str):
    if quantization == "none":
        return None, {}
    