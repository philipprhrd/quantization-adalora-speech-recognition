from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from peft import (
    AdaLoraConfig,
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class LoRAFineTuneConfig:
    model_id: str
    train_data_path: str
    output_dir: str

    # LoRA / AdaLoRA
    adaptive: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    # AdaLoRA only
    adalora_init_r: int = 12
    adalora_target_r: int = 8
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10

    # Training
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    fp16: bool = False
    bf16: bool = False

    # Data
    max_label_length: int = 256
    max_audio_seconds: Optional[float] = None


class WhisperDataCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungültiges JSON in {path}:{line_no}: {exc}") from exc
    return data


def _build_peft_config(cfg: LoRAFineTuneConfig):
    target_modules = cfg.target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ]

    if cfg.adaptive:
        return AdaLoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            init_r=cfg.adalora_init_r,
            target_r=cfg.adalora_target_r,
            tinit=cfg.adalora_tinit,
            tfinal=cfg.adalora_tfinal,
            deltaT=cfg.adalora_delta_t,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            inference_mode=False,
        )

    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )


def _load_base_model_and_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    return model, processor


def finetune_lora(cfg: LoRAFineTuneConfig) -> Dict[str, str]:
    """
    Führt LoRA/AdaLoRA-Finetuning auf einem ausgewählten ASR-Modell durch und exportiert:
      1) Adapter (immer)
      2) optional merged Modell (wenn möglich)

    Rückgabe:
        {
          "adapter_dir": "...",
          "merged_dir": "... | ''",
          "training_output_dir": "..."
        }
    """
    torch.manual_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Lade Basismodell und Processor: %s", cfg.model_id)
    model, processor = _load_base_model_and_processor(cfg.model_id)

    peft_cfg = _build_peft_config(cfg)
    model = get_peft_model(model, peft_cfg)

    # Enable grad checkpointing compatibility for some whisper variants
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    LOGGER.info("Lade Trainingsdaten aus: %s", cfg.train_data_path)
    train_dataset = ASRJsonlDataset(
        jsonl_path=cfg.train_data_path,
        processor=processor,
        max_label_length=cfg.max_label_length,
        max_audio_seconds=cfg.max_audio_seconds,
    )

    collator = WhisperDataCollator(processor)

    training_output_dir = output_dir / "trainer_output"
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"

    args = Seq2SeqTrainingArguments(
        output_dir=str(training_output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to=[],
        remove_unused_columns=False,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    LOGGER.info("Starte LoRA%s Fine-Tuning...", "/AdaLoRA" if cfg.adaptive else "")
    trainer.train()

    LOGGER.info("Speichere Adapter nach: %s", adapter_dir)
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    merged_path = ""
    try:
        LOGGER.info("Versuche Merge Adapter -> Basismodell: %s", merged_dir)
        peft_model = PeftModel.from_pretrained(
            AutoModelForSpeechSeq2Seq.from_pretrained(cfg.model_id),
            str(adapter_dir),
        )
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        merged_path = str(merged_dir)
    except Exception as exc:
        LOGGER.warning("Merge/Export des vollständigen Modells fehlgeschlagen: %s", exc)

    return {
        "adapter_dir": str(adapter_dir),
        "merged_dir": merged_path,
        "training_output_dir": str(training_output_dir),
    }
