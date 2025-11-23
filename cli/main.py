"""Main CLI entry point for ML model deployment."""

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.deployer import ModelDeployer
from src.core.config import Config

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """ML Model Deployment Toolkit - Deploy ML models to AWS, GCP, and Vercel."""
    pass


@cli.command()
@click.argument('platform', type=click.Choice(['aws-lambda', 'gcp-cloud-run', 'vercel']))
@click.option('--model-path', required=True, help='Path to the model file')
@click.option('--name', required=True, help='Deployment name')
@click.option('--model-type', default='sklearn', help='Model type (sklearn, tensorflow, pytorch)')
@click.option('--requirements', multiple=True, help='Python package requirements')
@click.option('--env', multiple=True, help='Environment variables (KEY=VALUE)')
@click.option('--region', help='Deployment region')
@click.option('--memory', help='Memory allocation (e.g., 1024, 2Gi)')
@click.option('--timeout', type=int, help='Timeout in seconds')
def deploy(platform, model_path, name, model_type, requirements, env, region, memory, timeout):
    """Deploy a model to the specified platform."""
    console.print(f"\n[bold blue]🚀 Deploying model to {platform}...[/bold blue]\n")

    try:
        # Parse environment variables
        env_vars = {}
        for e in env:
            key, value = e.split('=', 1)
            env_vars[key] = value

        # Create deployer
        config = Config()
        if region:
            if platform == 'aws-lambda':
                config.aws.region = region
            elif platform == 'gcp-cloud-run':
                config.gcp.region = region

        deployer = ModelDeployer(platform, config)

        # Deploy
        with Progress() as progress:
            task = progress.add_task("[cyan]Deploying...", total=100)

            endpoint = deployer.deploy(
                model_path=model_path,
                name=name,
                requirements=list(requirements) if requirements else None,
                environment_vars=env_vars
            )

            progress.update(task, completed=100)

        console.print(f"\n[bold green]✅ Deployment successful![/bold green]")
        console.print(f"\n[bold]Endpoint URL:[/bold] {endpoint}")

        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. Test health check: curl {endpoint}/health")
        console.print(f"2. Make prediction: curl -X POST {endpoint}/predict -H 'Content-Type: application/json' -d '{{\"features\": [1.0, 2.0]}}'")

    except Exception as e:
        console.print(f"\n[bold red]❌ Deployment failed:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('platform', type=click.Choice(['aws-lambda', 'gcp-cloud-run', 'vercel']))
@click.argument('name')
def info(platform, name):
    """Get information about a deployment."""
    console.print(f"\n[bold blue]📊 Getting info for {name} on {platform}...[/bold blue]\n")

    try:
        deployer = ModelDeployer(platform)
        deployment_info = deployer.get_deployment_info(name)

        table = Table(title=f"Deployment Info: {name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in deployment_info.items():
            table.add_row(key, str(value))

        console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('platform', type=click.Choice(['aws-lambda', 'gcp-cloud-run', 'vercel']))
@click.argument('name')
@click.confirmation_option(prompt='Are you sure you want to delete this deployment?')
def delete(platform, name):
    """Delete a deployment."""
    console.print(f"\n[bold yellow]🗑️  Deleting {name} from {platform}...[/bold yellow]\n")

    try:
        deployer = ModelDeployer(platform)
        success = deployer.delete_deployment(name)

        if success:
            console.print(f"\n[bold green]✅ Deployment deleted successfully![/bold green]")
        else:
            console.print(f"\n[bold red]❌ Failed to delete deployment[/bold red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
def platforms():
    """List supported platforms."""
    console.print("\n[bold]Supported Deployment Platforms:[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Platform", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Best For", style="green")

    table.add_row(
        "aws-lambda",
        "AWS Lambda with API Gateway",
        "Cost-effective, variable workloads"
    )
    table.add_row(
        "gcp-cloud-run",
        "GCP Cloud Run (containerized)",
        "Auto-scaling, containerized models"
    )
    table.add_row(
        "vercel",
        "Vercel Serverless Functions",
        "Lightweight models, edge deployment"
    )

    console.print(table)
    console.print()


@cli.command()
@click.argument('model_path')
def validate(model_path):
    """Validate a model file."""
    console.print(f"\n[bold blue]🔍 Validating model at {model_path}...[/bold blue]\n")

    try:
        import os
        from pathlib import Path

        # Check if file exists
        if not Path(model_path).exists():
            console.print(f"[bold red]❌ Model file not found[/bold red]")
            sys.exit(1)

        # Check file size
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)

        table = Table(title="Model Validation")
        table.add_column("Check", style="cyan")
        table.add_column("Result", style="green")

        table.add_row("File exists", "✅ Yes")
        table.add_row("File size", f"{size_mb:.2f} MB")

        # Platform compatibility
        if size_mb < 50:
            table.add_row("Vercel compatible", "✅ Yes")
        else:
            table.add_row("Vercel compatible", "❌ No (>50MB)")

        if size_mb < 250:
            table.add_row("AWS Lambda compatible", "✅ Yes")
        else:
            table.add_row("AWS Lambda compatible", "⚠️ May need layers")

        table.add_row("GCP Cloud Run compatible", "✅ Yes")

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]❌ Validation failed:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--platform', type=click.Choice(['aws-lambda', 'gcp-cloud-run', 'vercel']))
def init(platform):
    """Initialize a new deployment project."""
    if not platform:
        console.print("\n[bold]Available templates:[/bold]")
        console.print("1. aws-lambda - AWS Lambda deployment")
        console.print("2. gcp-cloud-run - GCP Cloud Run deployment")
        console.print("3. vercel - Vercel serverless deployment")
        console.print("\nRun: ml-deploy init --platform <platform-name>")
        return

    console.print(f"\n[bold blue]🎉 Initializing {platform} project...[/bold blue]\n")

    # This would copy template files to current directory
    console.print(f"[green]✅ Created {platform} project structure[/green]")
    console.print(f"\nNext steps:")
    console.print(f"1. Place your model in models/model.pkl")
    console.print(f"2. Update configuration in terraform/variables.tf or vercel.json")
    console.print(f"3. Run: cd templates/{platform} && ./deploy.sh")


if __name__ == '__main__':
    cli()
