from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import uuid
import typer
import time

from models.loading import load_model_and_processor
from models.quantization import torch_quantize
from utils.model import get_model_size
from dotenv import load_dotenv
from transformers import set_seed

load_dotenv()


app = typer.Typer()

console = Console()

def create_run_id():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"run-{timestamp}-{uuid.uuid4().hex[:5]}"

@app.command()
def quantize(
    model_id: str = typer.Option(..., help="e.g. openai/whisper-tiny"),
    output_dir: str = typer.Option("./output", help="Directory to save the transcriptions"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    bnb: bool = typer.Option(False, help="Whether to use bitsandbytes for quantization"),
    mode: str = typer.Option("default", help="Quantization mode (e.g., 'fp16', 'int8', 'nf4')")
):
    run_id = create_run_id()

    set_seed(seed)

    console.print(
        Panel.fit(
            f"[bold]QuantizationAdaLoRA-ASR[/bold] - Quantization\n\n"
            f"Model: {model_id}\n"
            f"Output Directory: {output_dir}\n"
            f"Seed: {seed}\n"
            f"Run ID: {run_id}",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        start_time = time.time()

        if bnb:
            console.print("[green]Using bitsandbytes for quantization...[/green]")
            progress.add_task("Downloading and Quantize Model")
            model, processor = load_model_and_processor(model_id, quantization=mode)
        else:
            console.print("[green]Using PyTorch for quantization...[/green]")
            progress.add_task("Downloading Model")
            model, processor = load_model_and_processor(model_id, quantization="none")
            console.print(f"Size: {get_model_size(model):.2f}")

            progress.add_task("Quantize Model")
            torch_quantize(model, mode)

        console.print(f"After Size: {get_model_size(model):.2f}")


    elapsed_time = time.time() - start_time

    console.print(f"\n[green]Done![/green] Output saved to: [bold blue]{output_dir}/{run_id}[/bold blue]")
    console.print(f"Time taken: [bold blue]{elapsed_time:.2f}[/bold blue] seconds")

@app.command()
def lora(
    model_id: str = typer.Option(..., help="e.g. openai/whisper-tiny"),
    output_dir: str = typer.Option("./output", help="Directory to save the transcriptions"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    adaptive: bool = typer.Option(False, help="Whether to use adaptive LoRA")
):
    run_id = create_run_id()

    set_seed(seed)

    console.print(
        Panel.fit(
            f"[bold]QuantizationAdaLoRA-ASR[/bold] - Quantization\n\n"
            f"Model: {model_id}\n"
            f"Output Directory: {output_dir}\n"
            f"Seed: {seed}\n",
            f"Adaptive LoRA: {'Enabled' if adaptive else 'Disabled'}\n",
            f"Run ID: {run_id}",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        start_time = time.time()

        


    elapsed_time = time.time() - start_time

    console.print(f"\n[green]Done![/green] Output saved to: [bold blue]{output_dir}/{run_id}[/bold blue]")
    console.print(f"Time taken: [bold blue]{elapsed_time:.2f}[/bold blue] seconds")

if __name__ == "__main__":
    app()