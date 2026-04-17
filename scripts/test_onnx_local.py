"""
Local ONNX model tester.

Two modes:
  --audio-file   Transcribe a single audio file and print the result.
  --dataset-path Run full evaluation on the test split (WER, CER, latency).

Usage:
    # Single file
    python scripts/test_onnx_local.py \
        --model-path runs/baseline_onnx/onnx_fp32 \
        --audio-file path/to/audio.wav

    # Full dataset evaluation
    python scripts/test_onnx_local.py \
        --model-path runs/adalora_int8_onnx/onnx_int8 \
        --dataset-path data/cv_de_eval
"""

import argparse
import json
import time

import numpy as np
import soundfile as sf
import torch
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor

from src.evaluation.evaluate_onnx import OnnxModelEvaluator


def transcribe_file(model_path: str, audio_path: str) -> None:
    print(f"Loading ONNX model from {model_path} ...")
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_path, provider="CPUExecutionProvider"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_tensor = inputs.get("input_features") or inputs.get("input_values")

    t0 = time.perf_counter()
    predicted_ids = model.generate(input_tensor)
    elapsed = time.perf_counter() - t0

    text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    duration = len(audio) / sample_rate

    print(f"\nTranscription : {text}")
    print(f"Inference time: {elapsed:.3f}s")
    print(f"Audio duration: {duration:.3f}s")
    print(f"RTF           : {elapsed / duration:.3f}")


def evaluate_dataset(model_path: str, dataset_path: str) -> None:
    evaluator = OnnxModelEvaluator(model_path)
    results = evaluator.evaluate_dataset(dataset_path)
    print("\nResults JSON:")
    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test an ONNX model locally.")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio-file",   help="Single .wav/.mp3 file to transcribe.")
    group.add_argument("--dataset-path", help="datasets.load_from_disk path for full eval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audio_file:
        transcribe_file(args.model_path, args.audio_file)
    else:
        evaluate_dataset(args.model_path, args.dataset_path)


if __name__ == "__main__":
    main()
