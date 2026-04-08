from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi


def main() -> int:
    token = os.getenv("HF_TOKEN")
    space_id = os.getenv("HF_SPACE_ID")
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")

    if not token:
        print("HF_TOKEN is required", file=sys.stderr)
        return 1
    if not space_id:
        print("HF_SPACE_ID is required (format: username/space-name)", file=sys.stderr)
        return 1
    if not api_base_url:
        print("API_BASE_URL is required", file=sys.stderr)
        return 1
    if not model_name:
        print("MODEL_NAME is required", file=sys.stderr)
        return 1

    api = HfApi(token=token)

    # Keep names exactly as required by evaluator instructions.
    api.add_space_secret(repo_id=space_id, key="HF_TOKEN", value=token)
    api.add_space_variable(repo_id=space_id, key="API_BASE_URL", value=api_base_url)
    api.add_space_variable(repo_id=space_id, key="MODEL_NAME", value=model_name)

    print(f"Updated Space secrets/variables for {space_id}")
    print("  secret: HF_TOKEN")
    print("  variable: API_BASE_URL")
    print("  variable: MODEL_NAME")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
