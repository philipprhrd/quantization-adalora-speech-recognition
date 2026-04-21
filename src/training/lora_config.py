from peft import AdaLoraConfig, LoraConfig, TaskType


def _get_target_modules(base_model: str) -> list[str]:
    if "whisper" in base_model:
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    if "moonshine" in base_model:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    raise ValueError(f"Unsupported base model for LoRA: {base_model}")


def get_adalora_config(base_model: str, tinit: int, tfinal: int, total_step: int) -> LoraConfig:
    target_modules = _get_target_modules(base_model)

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


def get_lora_config(base_model: str, r: int = 8) -> LoraConfig:
    target_modules = _get_target_modules(base_model)

    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=target_modules,
        r=r,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
