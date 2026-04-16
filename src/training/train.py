import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
from datasets import load_from_disk
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
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
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

        model_kwargs = {
            "torch_dtype": torch.float16
            if torch.cuda.is_available()
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

        
        print("Applying LoRA configuration")
        lora_config = get_adalora_config(model_name)
        if self.quantization is not None:
            self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        # Moonshine ist kein Whisper-Derivat und verwendet keine Whisper-Sprach-/Task-Prompts in der GenerationConfig.
        if self.model.config.model_type == "whisper":
            self.model.generation_config.language = "german"
            self.model.generation_config.task = "transcribe"

        self.model.generation_config.forced_decoder_ids = None

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
        eval_steps: int = 300,
        save_steps: int = 300,
    ) -> None:
        print(f"Loading data from {dataset_path}")
        dataset = load_from_disk(dataset_path)

        # Compute total training steps so AdaLoRA's rank scheduler has the
        # correct budget.  Must happen after dataset loading (size is unknown
        # until here) and before Trainer.train().
        steps_per_epoch = math.ceil(
            len(dataset["train"]) / (batch_size * gradient_accumulation_steps)
        )
        total_steps = steps_per_epoch * num_epochs
        print(f"AdaLoRA total_step: {total_steps} ({steps_per_epoch} steps/epoch × {num_epochs} epochs)")

        for config in self.model.peft_config.values():
            config.total_step = total_steps
        if hasattr(self.model.base_model, "rankallocator"):
            self.model.base_model.rankallocator.total_step = total_steps

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
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            label_names=["labels"],
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
        )

        print("Starting training...")
        trainer.train(resume_from_checkpoint=True)

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

        
        self.model.save_pretrained(output_dir)

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
