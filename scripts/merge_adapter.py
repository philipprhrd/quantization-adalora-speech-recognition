"""
Merge a PEFT adapter into a full-precision (fp32) base model.

Needed for pipelines 1 & 2 (int4/int8 → AdaLoRA → eval):
  train.py skips the merge when bitsandbytes quantization was used during
  training, because the base model is in a quantized format that cannot be
  merged directly.  This script loads the base model fresh in fp32 and then
  merges the adapter on top.

For pipelines 3 & 4 (AdaLoRA → eval) the merge is already done by train.py
and saved to <output_dir>/merged — you can skip this script.

Usage:
    python scripts/merge_adapter.py \\
        --base-model openai/whisper-tiny \\
        --adapter-path runs/int4_adalora \\
        --output-dir runs/int4_adalora_merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter into the base model (fp32)."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="HuggingFace model name or local path to the base model.",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to the directory that contains the PEFT adapter weights.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to save the merged fp32 model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading base model in fp32: {args.base_model}")
    base = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.base_model,
        dtype=torch.float32,
    )

    print(f"Loading PEFT adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged.save_pretrained(args.output_dir)

    processor = AutoProcessor.from_pretrained(args.adapter_path)
    processor.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
