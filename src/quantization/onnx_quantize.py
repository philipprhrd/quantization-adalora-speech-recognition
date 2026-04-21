"""
ONNX export and quantization for CPU inference.

Three functions, called in order:
  1. export_to_onnx   – HuggingFace model  →  ONNX files
  2. quantize_int8    – ONNX files         →  INT8-quantized ONNX files
  3. quantize_int4    – ONNX files         →  INT4-quantized ONNX files

Each function saves its output to a separate directory so you can inspect
every step independently.
"""

import shutil
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor


def export_to_onnx(
    model_path: str, output_dir: str, processor_path: str | None = None
) -> None:
    """
    Export a HuggingFace speech model to ONNX.

    optimum creates one ONNX file per sub-graph:
      encoder_model.onnx
      decoder_model.onnx
      decoder_with_past_model.onnx
    plus the usual tokenizer / config files.

    processor_path: optional separate source for the processor/tokenizer.
        Needed for models with custom tokenizer backends (e.g. Moonshine's
        TokenizersBackend), whose remote-code files are lost during
        processor.save_pretrained(). Point this at the Hub base model name
        (e.g. 'usefulsensors/moonshine-tiny') to load the processor fresh.
        Defaults to model_path.
    """
    print(f"Exporting to ONNX: {model_path}")
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path, export=True)
    model.save_pretrained(output_dir)

    src = processor_path if processor_path is not None else model_path
    print(f"Loading processor from: {src}")
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    processor.save_pretrained(output_dir)

    print(f"ONNX model saved to {output_dir}")


def quantize_int8(onnx_dir: str, output_dir: str) -> None:
    """
    Apply dynamic INT8 quantization to every .onnx file in onnx_dir.

    Dynamic quantization: weights are quantized offline, activations at
    runtime. No calibration data needed. Works on CPU.
    """
    onnx_dir = Path(onnx_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    print(f"INT8 quantizing {len(onnx_files)} ONNX file(s) in {onnx_dir}")

    for f in onnx_files:
        out = output_dir / f.name
        print(f"  {f.name} ...")
        quantize_dynamic(
            str(f),
            str(out),
            weight_type=QuantType.QInt8,
        )

    # Copy tokenizer / config files unchanged
    for f in onnx_dir.iterdir():
        if f.suffix != ".onnx" and not (output_dir / f.name).exists():
            shutil.copy2(f, output_dir / f.name)

    print(f"INT8 model saved to {output_dir}")


def quantize_int4(onnx_dir: str, output_dir: str) -> None:
    """
    Apply NF4 weight quantization to every .onnx file in onnx_dir.

    Uses MatMulBnb4Quantizer with NF4 — the same 4-bit format as
    bitsandbytes' bnb_4bit_quant_type="nf4".
    Only MatMul weight tensors are quantized; everything else stays in FP32.
    """
    from onnxruntime.quantization.matmul_bnb4_quantizer import MatMulBnb4Quantizer

    onnx_dir = Path(onnx_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    print(f"INT4 (NF4) quantizing {len(onnx_files)} ONNX file(s) in {onnx_dir}")

    for f in onnx_files:
        out = output_dir / f.name
        print(f"  {f.name} ...")
        model = onnx.load(str(f))
        quant = MatMulBnb4Quantizer(model, MatMulBnb4Quantizer.NF4, block_size=32)
        quant.process()
        onnx.save(quant.model.model, str(out))

    # Copy tokenizer / config files unchanged
    for f in onnx_dir.iterdir():
        if f.suffix != ".onnx" and not (output_dir / f.name).exists():
            shutil.copy2(f, output_dir / f.name)

    print(f"INT4 model saved to {output_dir}")
