import torch
import torch.nn as nn
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int4WeightOnlyConfig,
    quantize_
)
from transformers import BitsAndBytesConfig

# Integer‑Quantisierung
# int8_dynamic_activation_int8_weight (W8A8, dynamic activations)
# int8_weight_only (nur Gewichte int8)
# int4_weight_only (nur Gewichte int4, gruppenweise)
# int8_dynamic_activation_int4_weight (int8‑Aktivierungen, int4‑Gewichte)
# Float8‑Quantisierung
# float8_weight_only (Gewichte in Float8, Berechnung in höherer Präzision)
# float8_dynamic_activation_float8_weight
# float8_static_activation_float8_weight
# (teils mit Varianten wie e4m3/e5m2)

def build_quant_config(quantization: str) -> BitsAndBytesConfig:
    """
    Build quantization config for BitsAndBytes.
    
    Args:
        quantization: "int8", "int4", "nf4", or "none"
    
    Returns:
        tuple: (quantization_config, extra_kwargs)
    """
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    
    elif quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="int4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    elif quantization == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Default: no quantization
    return None, {}


def torch_quantize(model: nn.Module, quantization: str):
    """
    Quantize model using PyTorch's native quantization.
    
    Args:
        model: The model to quantize
        quantization: "int8", "int4", "nf4", "fp16", or "none"
    
    Returns:
        The quantized model
    """
    if quantization == "int8":
        # W8A8: int8 weights + dynamic int8 activations
        config = Int8DynamicActivationInt8WeightConfig()
        return quantize_(model, config)

    elif quantization == "int4":
        # Gewicht-only int4 (W4), Aktivierungen bleiben in höherer Präzision
        config = Int4WeightOnlyConfig(
            group_size=128
        )
        return quantize_(model, config)
    
    # "none" or unsupported: return model unchanged
    return model
