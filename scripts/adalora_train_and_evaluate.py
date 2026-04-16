import argparse
import json
from pathlib import Path

import torch

from src.evaluation.evaluate import ModelEvaluator
from src.training.train import ModelTrainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a speech model with AdaLoRA and evaluate it on the eval split."
    )

    # --- Model ---
    parser.add_argument(
        "--model-name",
        required=True,
        help="Hugging Face model name or local path to the base model.",
    )

    # --- Data ---
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to a datasets.load_from_disk dataset with 'train' and 'eval' splits.",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the trained adapter and checkpoints are saved.",
    )

    # --- Training hyperparameters ---
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device train/eval batch size (default: 8).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Number of warmup steps (default: 200).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Log every N steps (default: 100).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Run evaluation every N steps (default: 500).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500).",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed-precision (fp16) training.",
    )

    # --- Device / quantization (shared for training and evaluation) ---
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for training: 'auto', 'cuda', 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "int4", "int8"],
        help="Optional bitsandbytes quantization for the base model (default: none).",
    )

    # --- Evaluation-only options ---
    parser.add_argument(
        "--eval-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used for evaluation (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--eval-quantization",
        default=None,
        choices=["none", "int4", "int8"],
        help=(
            "Quantization to use during evaluation. "
            "Defaults to --quantization when not set."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_quantization = None if args.quantization == "none" else args.quantization

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("AdaLoRA Training")
    print("=" * 60)

    trainer = ModelTrainer(
        model_name=args.model_name,
        device=args.eval_device,
        quantization=train_quantization,
    )

    trainer.train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=not args.no_fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )

    if args.skip_eval:
        print("\nSkipping evaluation (--skip-eval was set).")
        return

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("AdaLoRA Evaluation")
    print("=" * 60)

    # Resolve which quantization to use for evaluation.
    # --eval-quantization overrides --quantization when explicitly provided.
    if args.eval_quantization is not None:
        eval_quantization = None if args.eval_quantization == "none" else args.eval_quantization
    else:
        eval_quantization = train_quantization

    evaluator = ModelEvaluator(
        model_path=args.output_dir,
        base_model=args.model_name,
        is_lora=True,
        device=args.eval_device,
        quantization=eval_quantization,
    )

    results = evaluator.evaluate_dataset(args.dataset_path)

    print("\nResults JSON:")
    print(json.dumps(results, indent=2))

    with open(Path(args.output_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {Path(args.output_dir) / 'results.json'}")


if __name__ == "__main__":
    main()
