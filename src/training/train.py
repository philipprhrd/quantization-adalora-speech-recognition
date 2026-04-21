import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from datasets import load_from_disk
from jiwer import wer as compute_wer, cer as compute_cer
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from src.training.lora_config import get_adalora_config
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: AutoProcessor
    decoder_start_token_id: int | None

    def _resolve_input_name(
        self, feature: Dict[str, Union[List[int], torch.Tensor]]
    ) -> str:
        model_input_names = getattr(
            self.processor.feature_extractor, "model_input_names", []
        )
        if model_input_names and model_input_names[0] in feature:
            return model_input_names[0]

        # Moonshine trainiert auf rohen Audio-Werten (`input_values`), waehrend
        # Whisper-Datensaetze oft bereits Log-Mel-Features (`input_features`) enthalten.
        # Wir akzeptieren deshalb beide Formate und waehlen zur Laufzeit den passenden Key.
        for input_name in ("input_features", "input_values"):
            if input_name in feature:
                return input_name

        raise ValueError(
            "Dataset sample is missing a supported audio input column. "
            "Expected one of: input_features, input_values."
        )

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_name = self._resolve_input_name(features[0])
        input_features = [{input_name: feature[input_name]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True
        )
        # Keep only the audio tensor and its mask — removes any extra keys
        # (e.g. input_ids) that some feature extractors add and that would
        # conflict with the decoder's input_ids parameter.
        batch = {k: v for k, v in batch.items() if k in {input_name, "attention_mask"}}

        # Pad labels with -100 (loss ignore index) directly — avoids needing
        # a pad_token on the tokenizer, which Moonshine does not have by default.
        label_tensors = [
            torch.tensor(f["labels"], dtype=torch.long) for f in features
        ]
        labels = torch.nn.utils.rnn.pad_sequence(
            label_tensors, batch_first=True, padding_value=-100
        )

        if (
            self.decoder_start_token_id is not None
            and labels.shape[1] > 0
            and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class ModelTrainer:
    def __init__(
        self,
        model_name,
        device: str = "auto",
        quantization: str | None = None,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
    ):
        self.model_name = model_name
        self.quantization = quantization

        if self.quantization not in {None, "int4", "int8"}:
            raise ValueError("quantization must be one of: None, 'int4', 'int8'")

        if self.quantization is not None and not torch.cuda.is_available():
            raise ValueError(
                "bitsandbytes int4/int8 quantization requires a CUDA-enabled PyTorch environment."
            )

        self.processor = AutoProcessor.from_pretrained(model_name)

        print(f"Loading model: {model_name}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if bnb_4bit_compute_dtype not in dtype_map:
            raise ValueError(
                "bnb_4bit_compute_dtype must be one of: 'float16', 'bfloat16', 'float32'"
            )

        # float16 only for quantized models (bitsandbytes compute dtype);
        # non-quantized training uses float32 weights + fp16=True AMP in trainer.
        model_kwargs = {
            "torch_dtype": torch.float16
            if (torch.cuda.is_available() and self.quantization is not None)
            else torch.float32,
        }

        if self.quantization is not None:
            # Die bitsandbytes-Quantisierung muss direkt beim Laden des Basismodells
            # passieren, damit AdaLoRA auf dem bereits quantisierten Modell aufsetzt.
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
        elif self.quantization is not None:
            model_kwargs["device_map"] = {"": device}

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, **model_kwargs
        )

        if device != "auto" and self.quantization is None:
            self.model.to(device)

        if self.quantization is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        if self.processor.tokenizer.pad_token is None:
            pad_value = self.processor.tokenizer.eos_token or "<pad>"
            added = self.processor.tokenizer.add_special_tokens({"pad_token": pad_value})
            if added > 0:
                self.model.resize_token_embeddings(len(self.processor.tokenizer))

        # get_peft_model is called in train() once total_steps is known

    @staticmethod
    def _extract_suffix_number(path: Path, prefix: str) -> int | None:
        if not path.is_dir() or not path.name.startswith(prefix):
            return None

        suffix = path.name[len(prefix):]
        if not suffix.isdigit():
            return None

        return int(suffix)

    def _find_resume_checkpoint(self, output_dir: str) -> Path | None:
        output_path = Path(output_dir)

        print(f"[resume-debug] Scanning output_dir: {output_path.resolve()}")
        if not output_path.exists():
            print("[resume-debug] output_dir does not exist")
        else:
            entries = sorted(output_path.rglob("*"))
            if not entries:
                print("[resume-debug] output_dir is empty")
            for entry in entries:
                rel = entry.relative_to(output_path)
                depth = len(rel.parts) - 1
                indent = "  " * depth
                if entry.is_dir():
                    print(f"[resume-debug] {indent}{rel}/")
                else:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = -1
                    print(f"[resume-debug] {indent}{rel} ({size} bytes)")

        retry_dirs = [
            retry_dir
            for retry_dir in output_path.iterdir()
            if self._extract_suffix_number(retry_dir, "retry_") is not None
        ] if output_path.exists() else []

        if retry_dirs:
            latest_retry_dir = max(
                retry_dirs,
                key=lambda path: self._extract_suffix_number(path, "retry_"),
            )
            checkpoints = [
                checkpoint_dir
                for checkpoint_dir in latest_retry_dir.glob("checkpoint-*")
                if self._extract_suffix_number(checkpoint_dir, "checkpoint-") is not None
            ]
            if checkpoints:
                return max(
                    checkpoints,
                    key=lambda path: self._extract_suffix_number(path, "checkpoint-"),
                )
            return None

        checkpoints = [
            checkpoint_dir
            for checkpoint_dir in output_path.glob("checkpoint-*")
            if self._extract_suffix_number(checkpoint_dir, "checkpoint-") is not None
        ]
        if checkpoints:
            return max(
                checkpoints,
                key=lambda path: self._extract_suffix_number(path, "checkpoint-"),
            )

        return None

    def train(
        self,
        dataset_path: str,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        warmup_steps: int = 200,
        fp16: bool = True,
        gradient_accumulation_steps: int = 2,
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 1000,
        eval_samples: int | None = 500,
        generation_max_length: int = 225,
    ) -> None:
        print(f"Loading data from {dataset_path}")
        dataset = load_from_disk(dataset_path)
        eval_dataset = dataset["dev"]
        if eval_samples is not None:
            eval_dataset = eval_dataset.shuffle(seed=42, keep_in_memory=True).select(range(min(eval_samples, len(eval_dataset))))
            print(f"Using {len(eval_dataset)} eval samples (shuffled, seed=42)")

        train_dataset = dataset["train"]
        half_size = len(train_dataset) // 2
        train_dataset = train_dataset.shuffle(seed=42, keep_in_memory=True).select(range(half_size))
        print(f"Using {len(train_dataset)} train samples (50% subset, shuffled, seed=42)")

        # Compute total training steps so AdaLoRA's rank scheduler has the
        # correct budget.  Must happen after dataset loading (size is unknown
        # until here) and before Trainer.train().
        steps_per_epoch = math.ceil(
            len(train_dataset) / (batch_size * gradient_accumulation_steps)
        )
        total_steps = steps_per_epoch * num_epochs
        print(f"AdaLoRA total_step: {total_steps} ({steps_per_epoch} steps/epoch × {num_epochs} epochs)")

        tinit  = max(1, int(0.10 * total_steps))
        tfinal = max(tinit + 1, int(0.60 * total_steps))
        print(f"AdaLoRA schedule: tinit={tinit}, tfinal={tfinal}, total_step={total_steps}")

        print("Applying LoRA configuration")
        lora_config = get_adalora_config(self.model_name, tinit, tfinal, total_steps)
        self.model = get_peft_model(self.model, lora_config)

        # PeftModelForSeq2SeqLM is built for text encoders and always passes
        # input_ids=None and inputs_embeds=None through **kwargs. Audio models
        # (Whisper/Moonshine) don't have these as named params, so they leak
        # down to the decoder which has them as explicit params → conflict.
        _audio_base = self.model.base_model.model
        _orig_fwd = _audio_base.forward
        def _fwd_no_text_inputs(*args, **kwargs):
            kwargs.pop("input_ids", None)
            kwargs.pop("inputs_embeds", None)
            kwargs.pop("num_items_in_batch", None)
            return _orig_fwd(*args, **kwargs)
        _audio_base.forward = _fwd_no_text_inputs

        # Seq2SeqTrainer passes the full batch (including labels) to generate()
        # during eval. Strip labels before forwarding to avoid kwarg validation.
        _orig_generate = self.model.generate
        def _generate_no_labels(*args, **kwargs):
            kwargs.pop("labels", None)
            return _orig_generate(*args, **kwargs)
        self.model.generate = _generate_no_labels

        if self.model.config.model_type == "whisper":
            self.model.generation_config.language = "german"
            self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None

        # Moonshine has no pad_token_id by default — generate() requires it
        if self.model.config.pad_token_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
            self.model.config.pad_token_id = pad_id
            self.model.generation_config.pad_token_id = pad_id

        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = np.where(pred.label_ids != -100, pred.label_ids, pad_id)
            pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {
                "wer": compute_wer(label_str, pred_str),
                "cer": compute_cer(label_str, pred_str),
            }

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            fp16=fp16 and torch.cuda.is_available(),
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            logging_steps=logging_steps,
            predict_with_generate=True,
            generation_max_length=generation_max_length,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            remove_unused_columns=False,
            label_names=["labels"],
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
            compute_metrics=compute_metrics,
        )

        print("Starting training...")
        resume = self._find_resume_checkpoint(output_dir)
        if resume:
            print(f"Resuming from checkpoint: {resume}")
        else:
            print("No checkpoint found to resume from. Starting from scratch.")
        trainer.train(resume_from_checkpoint=resume)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # save training history
        history_path = output_path / "training_history.json"
        with history_path.open("w", encoding="utf-8") as history_file:
            json.dump(trainer.state.log_history, history_file, indent=2)

        print(f"Saved training history to {history_path}")

        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        if self.quantization is None:
            merged_output_dir = output_path / "merged"
            print(
                f"Merging LoRA adapter into base model and saving to {merged_output_dir}"
            )
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(merged_output_dir)
            self.processor.save_pretrained(merged_output_dir)
        else:
            print(
                "Skipping merged model export because the base model was loaded with bitsandbytes quantization. "
                "Save the adapter and merge it later into a full-precision base model if needed."
            )

        print("Training complete!")


if __name__ == "__main__":
    pass
