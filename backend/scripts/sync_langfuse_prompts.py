"""Script to deploy/sync managed prompts to Langfuse."""
import sys
from pathlib import Path

# Ensure backend root is on sys.path
_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR.parent))

from backend.core.langfuse import get_langfuse_client, _FALLBACK_PROMPTS

def main():
    client = get_langfuse_client()
    if not client:
        print("[sync] Langfuse client is not available.", file=sys.stderr)
        return

    for name, prompt_text in _FALLBACK_PROMPTS.items():
        try:
            client.create_prompt(
                name=name,
                prompt=prompt_text,
                labels=["production"],
                type="text",
            )
            print(f"[sync] Successfully created/updated production prompt: '{name}'")
        except Exception as exc:
            print(f"[sync] Failed to create prompt '{name}': {exc}", file=sys.stderr)

if __name__ == "__main__":
    main()
