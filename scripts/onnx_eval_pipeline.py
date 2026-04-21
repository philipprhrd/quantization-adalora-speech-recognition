"""
Export → Quantize → Evaluate on CPU.

Takes a merged HuggingFace model (fp32) and runs it through three steps:
  1. Export to ONNX
  2. Quantize to INT8 or INT4 (optional)
  3. Evaluate on CPU and write results.json

This covers the eval side of all four pipelines:

  Pipeline 1: int4  → AdaLoRA train → merge_adapter.py → this script --quantization int8
  Pipeline 2: int8  → AdaLoRA train → merge_adapter.py → this script --quantization int8
  Pipeline 3: AdaLoRA train → <output>/merged          → this script --quantization int4
  Pipeline 4: AdaLoRA train → <output>/merged          → this script --quantization int8

Each intermediate directory (onnx_fp32/, onnx_int8/ or onnx_int4/) is kept
on disk so you can re-run individual steps without redoing everything.

Usage:
    python scripts/onnx_eval_pipeline.py \\
        --model-path  runs/adalora_merged \\
        --dataset-path data/cv_de_eval \\
        --output-dir  runs/adalora_int8_onnx \\
        --quantization int8
"""

import argparse
import json
from pathlib import Path

from src.evaluation.evaluate_onnx import OnnxModelEvaluator
from src.quantization.onnx_quantize import export_to_onnx, quantize_int4, quantize_int8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a merged HF model to ONNX, quantize, and evaluate on CPU."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a merged HuggingFace model (fp32, not a PEFT adapter).",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to a datasets.load_from_disk dataset with an 'eval' split.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for ONNX files and results.json.",
    )
    parser.add_argument(
        "--quantization",
        required=True,
        choices=["none", "int8", "int4"],
        help="Quantization to apply after ONNX export.",
    )
    parser.add_argument(
        "--processor-path",
        default=None,
        help=(
            "Optional separate source for the processor/tokenizer. "
            "Needed for Moonshine (custom TokenizersBackend gets lost on save_pretrained) — "
            "pass the base HF name, e.g. 'usefulsensors/moonshine-tiny'. "
            "Defaults to --model-path."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Step 1: Export to ONNX (fp32)
    # ------------------------------------------------------------------
    onnx_fp32_dir = output_dir / "onnx_fp32"
    print("\n=== Step 1: Export to ONNX ===")
    export_to_onnx(args.model_path, str(onnx_fp32_dir), processor_path=args.processor_path)

    # ------------------------------------------------------------------
    # Step 2: Quantize
    # ------------------------------------------------------------------
    print(f"\n=== Step 2: Quantize ({args.quantization}) ===")
    if args.quantization == "none":
        eval_dir = onnx_fp32_dir
        print("Skipping quantization, evaluating fp32 ONNX model.")
    elif args.quantization == "int8":
        eval_dir = output_dir / "onnx_int8"
        quantize_int8(str(onnx_fp32_dir), str(eval_dir))
    else:
        eval_dir = output_dir / "onnx_int4"
        quantize_int4(str(onnx_fp32_dir), str(eval_dir))

    # ------------------------------------------------------------------
    # Step 3: Evaluate on CPU
    # ------------------------------------------------------------------
    print("\n=== Step 3: Evaluate on CPU ===")
    evaluator = OnnxModelEvaluator(str(eval_dir))
    results = evaluator.evaluate_dataset(args.dataset_path)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
