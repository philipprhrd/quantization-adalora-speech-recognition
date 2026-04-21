from dataclasses import dataclass
from transformers import AutoProcessor, WhisperProcessor
import torch


@dataclass
class DataCollatorMoonshineSeq2SeqWithPadding:
    processor: AutoProcessor
    decoder_start_token_id: int | None
    pad_token_id: int | None

    def __call__(self, features):
        # Extract input values and labels
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        # Pad audio inputs
        batch = self.processor.feature_extractor.pad(
            input_values,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Pad labels using PyTorch's efficient pad_sequence
        label_tensors = [torch.tensor(labels, dtype=torch.long) for labels in label_features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=-100  # Ignored in loss calculation
        )

        # Create decoder_input_ids: [BOS, t1, t2, ..., tN]
        # Labels are: [t1, t2, ..., tN, EOS]
        # So decoder_input_ids = [BOS] + labels[:-1]
        decoder_input_tensors = [
            torch.tensor(
                [self.decoder_start_token_id] + labels[:-1],
                dtype=torch.long
            )
            for labels in label_features
        ]

        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        batch["decoder_input_ids"] = decoder_input_ids
        batch["labels"] = labels_padded

        return batch


@dataclass
class DataCollatorWhisperSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int | None

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
            return_attention_mask=True
        )

        batch = {k: v for k, v in batch.items() if k in {"input_features", "attention_mask"}}

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
