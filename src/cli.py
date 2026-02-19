from rich.console import Console
from rich.panel import Panel
import typer

app = typer.Typer()

console = Console()

@app.command()
def quantize(
    model: str = typer.Option(..., help="e.g. openai/whisper-tiny"),
    output_dir: str = typer.Option("./output", help="Directory to save the transcriptions"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    console.print(
        Panel.fit(
            f"[bold]QuantizationAdaLoRA-ASR[/bold] - Quantization\n\n"
            f"Model: {model}\n"
            f"Output Directory: {output_dir}\n"
            f"Seed: {seed}",
            border_style="blue",
        )
    )

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