from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer
import asyncio

from models.loading import load_model_and_processor
from models.quantization import torch_quantize

app = typer.Typer()

console = Console()

@app.command()
def quantize(
    model_id: str = typer.Option(..., help="e.g. openai/whisper-tiny"),
    output_dir: str = typer.Option("./output", help="Directory to save the transcriptions"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    bnb: bool = typer.Option(False, help="Whether to use bitsandbytes for quantization"),
    mode: str = typer.Option("default", help="Quantization mode (e.g., 'fp16', 'int8', 'nf4')")
):
    console.print(
        Panel.fit(
            f"[bold]QuantizationAdaLoRA-ASR[/bold] - Quantization\n\n"
            f"Model: {model_id}\n"
            f"Output Directory: {output_dir}\n"
            f"Seed: {seed}",
            border_style="blue",
        )
    )

    async def _run():
        if bnb:
            console.print("[green]Using bitsandbytes for quantization...[/green]")
            model, processor = await load_model_and_processor(model_id, quantization=mode)
        else:
            console.print("[green]Using PyTorch for quantization...[/green]")
            model, processor = await load_model_and_processor(model_id, quantization="none")
            model = await torch_quantize(model, mode)

    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task(description="Loading and quantizing model...", total=None)
        asyncio.run(_run())

    console.print(f"\n[green]Done![/green] Output saved to: [bold blue]{output_dir}[/bold blue]")
    # run id
    # time run

@app.command()
def lora(
    model: str = typer.Option(..., help="e.g. openai/whisper-tiny"),
    output_dir: str = typer.Option("./output", help="Directory to save the transcriptions"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    adaptive: bool = typer.Option(False, help="Whether to use adaptive LoRA"),
):
    print(f"Fine-tuning model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Adaptive LoRA: {adaptive}")

if __name__ == "__main__":
    app()