import io

import torch

def get_model_size(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()

    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / (1024 * 2)

    return size_mb