"""Generate API keys for authentication."""

import secrets
import hashlib
from datetime import datetime
from rich.console import Console

console = Console()


def generate_api_key(prefix: str = "ml_") -> tuple[str, str]:
    """
    Generate a new API key.

    Args:
        prefix: Prefix for the API key

    Returns:
        Tuple of (api_key, hashed_api_key)
    """
    # Generate random part
    random_part = secrets.token_urlsafe(32)
    api_key = f"{prefix}{random_part}"

    # Hash for storage
    hashed = hashlib.sha256(api_key.encode()).hexdigest()

    return api_key, hashed


def main():
    """Main entry point."""
    console.print("\n[bold blue]🔑 API Key Generator[/bold blue]\n")

    # Generate key
    api_key, hashed_key = generate_api_key()

    console.print("[green]✓[/green] Generated new API key\n")

    console.print("[bold]API Key (save this securely):[/bold]")
    console.print(f"[yellow]{api_key}[/yellow]\n")

    console.print("[bold]Hashed Key (store in database):[/bold]")
    console.print(f"[cyan]{hashed_key}[/cyan]\n")

    console.print("[bold]Environment Variable:[/bold]")
    console.print(f"[blue]API_KEY={api_key}[/blue]\n")

    console.print("[red]⚠️  Warning:[/red] Store the API key securely.")
    console.print("You won't be able to retrieve it again.\n")

    # Save to file
    with open(".api_keys.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()}: {api_key}\n")

    console.print("[green]✓[/green] API key saved to .api_keys.txt")
    console.print()


if __name__ == "__main__":
    main()
