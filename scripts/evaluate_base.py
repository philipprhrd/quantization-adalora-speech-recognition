import argparse
import json

import torch

from src.evaluation.evaluate import ModelEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a base speech model on the eval split of a dataset."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Hugging Face model name or local path to the base model.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to a datasets.load_from_disk dataset with an 'eval' split.",
    )
    parser.add_argument(
        "--eval-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used for evaluation.",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "int4", "int8"],
        help="Optional bitsandbytes quantization for GPU evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantization = None if args.quantization == "none" else args.quantization

    evaluator = ModelEvaluator(
        model_path=args.model_name,
        base_model=args.model_name,
        is_lora=False,
        device=args.device,
        quantization=quantization,
    )

    results = evaluator.evaluate_dataset(args.dataset_path)
    print("\nResults JSON:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
