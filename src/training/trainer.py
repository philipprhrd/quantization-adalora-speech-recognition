# ============================================
# Custom Trainer for Moonshine
# ============================================
import torch
from transformers import Seq2SeqTrainer

class MoonshineSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom trainer that properly handles Moonshine's generate() method.

    Moonshine uses variable-length sequences, so we need to:
    1. Calculate max_new_tokens based on audio duration
    2. Use phase-specific generation parameters
    """

    def __init__(self, *args, generation_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_config = generation_config or {}

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for Moonshine."""
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Calculate generation parameters based on audio length
        # Paper: Training instances in [4, 30] seconds
        if 'input_values' in inputs:
            audio_length = inputs['input_values'].shape[-1]
            audio_duration = audio_length / 16000  # 16kHz sampling rate
            # Roughly 6 tokens per second (conservative estimate)
            max_new_tokens = max(5, min(int(audio_duration * 6), 50))
        else:
            max_new_tokens = 50

        labels = inputs.get("labels", None)

        # If only computing loss, don't generate
        if prediction_loss_only:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else None
            return (loss, None, None)

        # Generate predictions
        with torch.no_grad():
            # Calculate loss first (if labels available)
            if has_labels:
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                loss = None

            # Generate transcriptions with phase-specific parameters
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                **self.generation_config
            }

            # Moonshine generate expects input_values (not input_features)
            generated_tokens = model.generate(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask", None),
                **generation_kwargs
            )

        if labels is not None:
            labels = labels.detach()

        return (loss, generated_tokens, labels)