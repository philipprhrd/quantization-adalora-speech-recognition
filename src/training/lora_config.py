from peft import AdaLoraConfig, LoraConfig, TaskType


def get_adalora_config(base_model: str, tinit: int, tfinal: int, total_step: int) -> LoraConfig:
    if "whisper" in base_model:
        target_modules = [
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "out_proj",  # Output projection (Whisper nennt es out_proj)
        ]
    elif "moonshine" in base_model:
        # Moonshine verwendet fuer die Attention-Ausgabe `o_proj` statt
        # Whispers `out_proj`, daher braucht es eigene LoRA-Targets.
        target_modules = [
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection (Moonshine nennt es o_proj, nicht out_proj)
            # fc1/fc2 (MLP-Layer) weglassen — Attention-Projektionen reichen für LoRA
        ]
    else:
        raise ValueError(f"Unsupported base model for LoRA: {base_model}")

    config = AdaLoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=target_modules,
        lora_alpha=16,
        lora_dropout=0.05,
        init_r=12,
        target_r=4,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=10,
        beta1=0.85,
        beta2=0.85,
        total_step=total_step
    )

    return config
