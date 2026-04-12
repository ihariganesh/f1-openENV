from __future__ import annotations

import os

import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


# Keep Space uploads minimal and evaluator-focused.
ALLOW_PATTERNS = [
    "Dockerfile",
    "README.md",
    "pyproject.toml",
    "openenv.yaml",
    "env.py",
    "models.py",
    "tasks.py",
    "inference.py",
    "validate-submission.sh",
    "app/**",
    "scripts/**",
    "src/**",
    "tests/**",
]

IGNORE_PATTERNS = [
    ".git/**",
    ".venv/**",
    ".vscode/**",
    ".pytest_cache/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    "*.pyo",
    "*.egg-info/**",
    "artifacts/**",
    "*.ipynb",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp",
    "*.gif",
    "*.mp4",
    "*.zip",
]


def main() -> int:
    token = os.getenv("HF_TOKEN")
    space_id = os.getenv("HF_SPACE_ID")

    if not token:
        print("HF_TOKEN is required", file=sys.stderr)
        return 1
    if not space_id:
        print("HF_SPACE_ID is required (format: username/space-name)", file=sys.stderr)
        return 1

    repo_dir = Path(__file__).resolve().parent.parent
    api = HfApi(token=token)

    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(repo_dir),
        token=token,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    print(f"Deployed to: https://{space_id.replace('/', '-')}.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
