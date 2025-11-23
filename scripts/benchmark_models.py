"""Benchmark script for comparing model performance."""

import time
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sklearn_model import SklearnModel
from rich.console import Console
from rich.table import Table

console = Console()


def benchmark_model(model, num_samples: int = 1000, num_features: int = 20):
    """
    Benchmark a model's performance.

    Args:
        model: Model instance
        num_samples: Number of samples to test
        num_features: Number of features per sample

    Returns:
        Dictionary of benchmark results
    """
    # Generate test data
    features = np.random.randn(num_samples, num_features)

    # Warm-up
    for _ in range(10):
        model.predict_with_preprocessing(features[0])

    # Single prediction benchmark
    single_times = []
    for i in range(100):
        start = time.time()
        model.predict_with_preprocessing(features[i])
        single_times.append((time.time() - start) * 1000)

    # Batch prediction benchmark
    batch_sizes = [1, 10, 50, 100, 500, 1000]
    batch_results = {}

    for batch_size in batch_sizes:
        if batch_size > num_samples:
            continue

        batch = features[:batch_size]

        start = time.time()
        model.predict_with_preprocessing(batch)
        elapsed = (time.time() - start) * 1000

        batch_results[batch_size] = {
            'total_time': elapsed,
            'per_item': elapsed / batch_size,
            'throughput': batch_size / (elapsed / 1000)
        }

    return {
        'single_prediction': {
            'mean': np.mean(single_times),
            'std': np.std(single_times),
            'min': np.min(single_times),
            'max': np.max(single_times),
            'p50': np.percentile(single_times, 50),
            'p95': np.percentile(single_times, 95),
            'p99': np.percentile(single_times, 99)
        },
        'batch_predictions': batch_results
    }


def main():
    """Main entry point."""
    console.print("\n[bold blue]🚀 Model Benchmarking[/bold blue]\n")

    # Load model
    model_path = Path(__file__).parent.parent / "examples/sklearn/models/sklearn_model.pkl"

    if not model_path.exists():
        console.print("[red]❌ Model not found. Train it first:[/red]")
        console.print("cd examples/sklearn && python train_model.py")
        return

    console.print(f"Loading model from {model_path}...")
    model = SklearnModel(str(model_path), "1.0.0")
    model.load()

    console.print("[green]✓[/green] Model loaded\n")

    # Run benchmark
    console.print("Running benchmarks...\n")
    results = benchmark_model(model)

    # Display single prediction results
    console.print("[bold]Single Prediction Performance[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Latency (ms)", style="green")

    single = results['single_prediction']
    table.add_row("Mean", f"{single['mean']:.2f}")
    table.add_row("Std Dev", f"{single['std']:.2f}")
    table.add_row("Min", f"{single['min']:.2f}")
    table.add_row("Max", f"{single['max']:.2f}")
    table.add_row("P50", f"{single['p50']:.2f}")
    table.add_row("P95", f"{single['p95']:.2f}")
    table.add_row("P99", f"{single['p99']:.2f}")

    console.print(table)
    console.print()

    # Display batch prediction results
    console.print("[bold]Batch Prediction Performance[/bold]\n")

    batch_table = Table(show_header=True, header_style="bold magenta")
    batch_table.add_column("Batch Size", style="cyan")
    batch_table.add_column("Total (ms)", style="green")
    batch_table.add_column("Per Item (ms)", style="yellow")
    batch_table.add_column("Throughput (items/s)", style="blue")

    for size, metrics in results['batch_predictions'].items():
        batch_table.add_row(
            str(size),
            f"{metrics['total_time']:.2f}",
            f"{metrics['per_item']:.2f}",
            f"{metrics['throughput']:.2f}"
        )

    console.print(batch_table)
    console.print()

    # Recommendations
    console.print("[bold]💡 Recommendations[/bold]\n")

    if single['p95'] < 10:
        console.print("[green]✓[/green] Excellent latency for real-time predictions")
    elif single['p95'] < 50:
        console.print("[yellow]⚠[/yellow] Good latency, suitable for most applications")
    else:
        console.print("[red]⚠[/red] Consider optimizing model for better latency")

    # Find optimal batch size
    best_batch = max(
        results['batch_predictions'].items(),
        key=lambda x: x[1]['throughput']
    )
    console.print(f"\n[blue]ℹ[/blue] Optimal batch size for throughput: {best_batch[0]}")
    console.print()


if __name__ == "__main__":
    main()
