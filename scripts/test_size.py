from datasets import load_from_disk
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a merged HF model to ONNX, quantize, and evaluate on CPU."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to a datasets.load_from_disk dataset with an 'eval' split.",
    )
    
    args = parser.parse_args()
    ds = load_from_disk(args.dataset_path)
    print(len(ds["train"][0]["input_values"]))
    print(len(ds["train"][1]["input_values"]))