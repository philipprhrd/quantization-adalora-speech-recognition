import time
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from jiwer import cer, wer


class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        base_model: str = "openai/whisper-tiny",
        is_lora: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        quantization: str | None = None,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
    ):
        self.quantization = quantization
        self.device = device

        if self.quantization not in {None, "int4", "int8"}:
            raise ValueError("quantization must be one of: None, 'int4', 'int8'")

        uses_bnb_quantization = self.quantization in {"int4", "int8"}

        if uses_bnb_quantization and not torch.cuda.is_available():
            raise ValueError(
                "bitsandbytes int4/int8 quantization requires a CUDA-enabled PyTorch environment."
            )

        if uses_bnb_quantization and device == "cpu":
            raise ValueError(
                "bitsandbytes int4/int8 quantization cannot be evaluated on CPU. "
                "Use quantization=None for full precision CPU inference."
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if bnb_4bit_compute_dtype not in dtype_map:
            raise ValueError(
                "bnb_4bit_compute_dtype must be one of: 'float16', 'bfloat16', 'float32'"
            )

        model_kwargs = {
            "dtype": torch.float16
            if torch.cuda.is_available()
            else torch.float32,
        }

        if uses_bnb_quantization:
            if self.quantization == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=dtype_map[bnb_4bit_compute_dtype],
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

        if device == "auto":
            model_kwargs["device_map"] = "auto"
        elif uses_bnb_quantization:
            model_kwargs["device_map"] = {"": device}

        if is_lora:
            print(f"Loading base model: {base_model}")
            base = AutoModelForSpeechSeq2Seq.from_pretrained(base_model, **model_kwargs)
            # Wenn der Adapter-Tokenizer durch das Training neue Special Tokens
            # dazubekommen hat (Moonshine: BOS/EOS/PAD), muss die Embedding-Tabelle
            # des Basismodells angepasst werden, sonst passt der Adapter nicht.
            adapter_vocab = len(self.processor.tokenizer)
            if base.get_input_embeddings().weight.shape[0] != adapter_vocab:
                print(
                    f"Resizing base embeddings {base.get_input_embeddings().weight.shape[0]}"
                    f" -> {adapter_vocab} to match adapter tokenizer"
                )
                base.resize_token_embeddings(adapter_vocab)
            print(f"Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            print(f"Loading model: {base_model}")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                base_model, **model_kwargs
            )

        # Moonshine ships without BOS/EOS/PAD and without decoder_start_token_id.
        # Required for generate() and to match how the model was trained.
        if self.model.config.model_type == "moonshine":
            added = self.processor.tokenizer.add_special_tokens({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
            })
            if added > 0:
                self.model.resize_token_embeddings(len(self.processor.tokenizer))
            self.model.config.bos_token_id = self.processor.tokenizer.bos_token_id
            self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.pad_token_id = self.model.config.pad_token_id
                self.model.generation_config.bos_token_id = self.model.config.bos_token_id
                self.model.generation_config.eos_token_id = self.model.config.eos_token_id
                self.model.generation_config.decoder_start_token_id = self.model.config.decoder_start_token_id

        if device != "auto" and not uses_bnb_quantization:
            self.model.to(device)

        self.model.eval()

        try:
            self.model_device = next(self.model.parameters()).device
        except StopIteration:
            self.model_device = torch.device(device if device != "auto" else "cpu")

    def model_size_mb(self) -> float:
        total_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.model.state_dict().values()
            if isinstance(tensor, torch.Tensor)
        )
        return total_bytes / (1024**2)

    def evaluate_dataset(self, dataset_path: str):
        print(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)["test"]

        predictions = []
        references = []
        inference_times = []
        rtf_values = []

        input_col = (
            "input_features"
            if "input_features" in dataset.column_names
            else "input_values"
        )

        print("Running evaluation...")
        for i, sample in enumerate(tqdm(dataset)):
            input_tensor = (
                torch.tensor(sample[input_col]).unsqueeze(0).to(self.model_device)
            )

            t0 = time.perf_counter()
            with torch.no_grad():
                # PeftModel.generate() erzwingt kwargs; Whisper erwartet
                # input_features, Moonshine input_values.
                predicted_ids = self.model.generate(**{input_col: input_tensor})
            # CUDA-Operationen sind asynchron — ohne synchronize() misst perf_counter()
            # nur den Kernel-Launch, nicht die echte Ausführungszeit auf der GPU.
            if self.model_device.type == "cuda":
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - t0

            # Rust-Tokenizer casten nach uint32 — negative IDs werfen OverflowError.
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                predicted_ids = torch.where(predicted_ids < 0, pad_id, predicted_ids)
            pred_text = self.processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            predictions.append(pred_text)
            references.append(sample["sentence"])

            # Erstes Sample überspringen: CUDA-Cache und Kernel-Compilation machen
            # es deutlich langsamer und würden Median/Std verzerren.
            if i > 0:
                inference_times.append(inference_time)
                rtf_values.append(inference_time / sample["audio_seconds"])

        metric_wer = float(wer(references, predictions))
        metric_cer = float(cer(references, predictions))
        inf_times = np.array(inference_times)
        rtf_arr = np.array(rtf_values)

        # model_size_mb() einmal berechnen und wiederverwenden statt zweimal aufrufen
        size_mb = self.model_size_mb()

        print(f"\nWER:                  {metric_wer:.4f}")
        print(f"CER:                  {metric_cer:.4f}")
        print(
            f"Inference time (med): {np.median(inf_times):.3f}s ± {np.std(inf_times):.3f}s"
        )
        print(f"Real time factor:     {np.median(rtf_arr):.3f} ± {np.std(rtf_arr):.3f}")
        print(f"Model size:           {size_mb:.1f} MB")

        return {
            "wer": metric_wer,
            "cer": metric_cer,
            "inference_time_median": float(np.median(inf_times)),
            "inference_time_std": float(np.std(inf_times)),
            "rtf_median": float(np.median(rtf_arr)),
            "rtf_std": float(np.std(rtf_arr)),
            "model_size_mb": size_mb,
        }


if __name__ == "__main__":
    evaluator = ModelEvaluator(
        model_path="openai/whisper-tiny",
        base_model="openai/whisper-tiny",
        is_lora=False,
    )

    evaluator.evaluate_dataset(dataset_path="")
