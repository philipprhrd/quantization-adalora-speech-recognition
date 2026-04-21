"""
ONNX model evaluator for CPU inference.

Mirrors the interface of ModelEvaluator in evaluate.py, but uses
ORTModelForSpeechSeq2Seq (optimum) instead of a PyTorch model.
This always runs on CPU; no CUDA required.
"""

import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from jiwer import cer, wer
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor
from tqdm import tqdm


class OnnxModelEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path

        print(f"Loading ONNX model from {model_path}")
        self.model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            provider="CPUExecutionProvider",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("ONNX model loaded on CPU")

    def model_size_mb(self) -> float:
        """Sum the on-disk size of all .onnx files (proxy for memory footprint)."""
        total = sum(
            f.stat().st_size for f in Path(self.model_path).glob("*.onnx")
        )
        return total / (1024 ** 2)

    def evaluate_dataset(self, dataset_path: str) -> dict:
        print(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)["test"]

        half_size = len(dataset) // 2
        dataset = dataset.shuffle(seed=42, keep_in_memory=True).select(range(half_size))
        print(f"Using {len(dataset)} test samples (50% subset, shuffled, seed=42)")

        input_col = (
            "input_features"
            if "input_features" in dataset.column_names
            else "input_values"
        )

        predictions = []
        references = []
        inference_times = []
        rtf_values = []

        print("Running evaluation on CPU...")
        for i, sample in enumerate(tqdm(dataset)):
            input_tensor = torch.tensor(sample[input_col]).unsqueeze(0)

            t0 = time.perf_counter()
            predicted_ids = self.model.generate(input_tensor)
            inference_time = time.perf_counter() - t0

            pred_text = self.processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            predictions.append(pred_text)
            references.append(sample["sentence"])

            # Skip first sample: cold-start overhead skews the median.
            if i > 0:
                inference_times.append(inference_time)
                rtf_values.append(inference_time / sample["audio_seconds"])

        metric_wer = float(wer(references, predictions))
        metric_cer = float(cer(references, predictions))
        inf_times = np.array(inference_times)
        rtf_arr = np.array(rtf_values)
        size_mb = self.model_size_mb()

        print(f"\nWER:                  {metric_wer:.4f}")
        print(f"CER:                  {metric_cer:.4f}")
        print(
            f"Inference time (med): {np.median(inf_times):.3f}s"
            f" ± {np.std(inf_times):.3f}s"
        )
        print(
            f"Real time factor:     {np.median(rtf_arr):.3f}"
            f" ± {np.std(rtf_arr):.3f}"
        )
        print(f"Model size (ONNX):    {size_mb:.1f} MB")

        return {
            "wer": metric_wer,
            "cer": metric_cer,
            "inference_time_median": float(np.median(inf_times)),
            "inference_time_std": float(np.std(inf_times)),
            "rtf_median": float(np.median(rtf_arr)),
            "rtf_std": float(np.std(rtf_arr)),
            "model_size_mb": size_mb,
        }
