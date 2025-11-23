"""Load testing script for deployed models."""

import asyncio
import aiohttp
import time
import json
import numpy as np
from typing import List, Dict
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()


async def make_prediction(
    session: aiohttp.ClientSession,
    endpoint: str,
    features: List[float],
    api_key: str = None
) -> Dict:
    """Make a single prediction request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {"features": features}

    start = time.time()
    try:
        async with session.post(
            f"{endpoint}/predict",
            json=data,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            latency = (time.time() - start) * 1000
            result = await response.json()

            return {
                "status": response.status,
                "latency": latency,
                "success": response.status == 200,
                "error": result.get("error") if response.status != 200 else None
            }
    except Exception as e:
        return {
            "status": 0,
            "latency": (time.time() - start) * 1000,
            "success": False,
            "error": str(e)
        }


async def run_load_test(
    endpoint: str,
    num_requests: int,
    concurrency: int,
    feature_size: int,
    api_key: str = None
):
    """
    Run load test against the endpoint.

    Args:
        endpoint: API endpoint URL
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
        feature_size: Size of feature vector
        api_key: Optional API key
    """
    console.print(f"\n[bold blue]🔥 Load Testing {endpoint}[/bold blue]\n")
    console.print(f"Requests: {num_requests}")
    console.print(f"Concurrency: {concurrency}")
    console.print(f"Feature size: {feature_size}\n")

    results = []

    async with aiohttp.ClientSession() as session:
        with Progress() as progress:
            task = progress.add_task("[cyan]Sending requests...", total=num_requests)

            # Create request batches
            batches = [
                list(range(i, min(i + concurrency, num_requests)))
                for i in range(0, num_requests, concurrency)
            ]

            for batch in batches:
                # Generate random features
                tasks = [
                    make_prediction(
                        session,
                        endpoint,
                        np.random.randn(feature_size).tolist(),
                        api_key
                    )
                    for _ in batch
                ]

                # Execute batch
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

                progress.update(task, advance=len(batch))

    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        latencies = [r["latency"] for r in successful]
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
    else:
        avg_latency = p50_latency = p95_latency = p99_latency = 0
        min_latency = max_latency = 0

    # Display results
    console.print("\n[bold green]📊 Results[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Requests", str(num_requests))
    table.add_row("Successful", str(len(successful)))
    table.add_row("Failed", str(len(failed)))
    table.add_row("Success Rate", f"{len(successful)/num_requests*100:.2f}%")
    table.add_row("", "")
    table.add_row("Avg Latency", f"{avg_latency:.2f} ms")
    table.add_row("Min Latency", f"{min_latency:.2f} ms")
    table.add_row("Max Latency", f"{max_latency:.2f} ms")
    table.add_row("P50 Latency", f"{p50_latency:.2f} ms")
    table.add_row("P95 Latency", f"{p95_latency:.2f} ms")
    table.add_row("P99 Latency", f"{p99_latency:.2f} ms")

    console.print(table)

    # Show errors if any
    if failed:
        console.print(f"\n[bold red]❌ Errors ({len(failed)})[/bold red]\n")
        error_counts = {}
        for r in failed:
            error = r.get("error", "Unknown error")
            error_counts[error] = error_counts.get(error, 0) + 1

        for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            console.print(f"  • {error}: {count}")

    console.print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load test ML model endpoint")
    parser.add_argument("endpoint", help="API endpoint URL")
    parser.add_argument("-n", "--requests", type=int, default=100,
                        help="Total number of requests (default: 100)")
    parser.add_argument("-c", "--concurrency", type=int, default=10,
                        help="Number of concurrent requests (default: 10)")
    parser.add_argument("-f", "--features", type=int, default=20,
                        help="Size of feature vector (default: 20)")
    parser.add_argument("--api-key", help="API key for authentication")

    args = parser.parse_args()

    # Run load test
    asyncio.run(run_load_test(
        args.endpoint,
        args.requests,
        args.concurrency,
        args.features,
        args.api_key
    ))


if __name__ == "__main__":
    main()
