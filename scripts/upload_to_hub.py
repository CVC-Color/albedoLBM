"""Upload the converted inference weights to the Hugging Face Hub.

Expects a local directory laid out as::

    weights/
    ├── albedo/model.safetensors
    └── shading/model.safetensors

The matching ``configs/albedo.yaml`` and ``configs/shading.yaml`` of this
repository are uploaded next to each ``model.safetensors`` as ``config.yaml``, and
``MODEL_CARD.md`` becomes the model card of the repository.

Usage::

    huggingface-cli login
    python scripts/upload_to_hub.py --weights_dir weights --repo_id davidserra9/albedo-lbm
"""

import argparse
import os

from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("albedo", "shading")


def main(args: argparse.Namespace) -> None:
    api = HfApi()

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repository ready: https://huggingface.co/{args.repo_id}")

    card = os.path.join(REPO_ROOT, "MODEL_CARD.md")
    if os.path.exists(card):
        api.upload_file(
            path_or_fileobj=card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message="Add model card",
        )
        print("Uploaded model card")

    for model in MODELS:
        weights = os.path.join(args.weights_dir, model, "model.safetensors")
        config = os.path.join(REPO_ROOT, "configs", f"{model}.yaml")
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"{weights} not found. Run scripts/convert_lightning_ckpt.py first."
            )

        api.upload_file(
            path_or_fileobj=config,
            path_in_repo=f"{model}/config.yaml",
            repo_id=args.repo_id,
            commit_message=f"Add {model} config",
        )
        size_gb = os.path.getsize(weights) / 1e9
        print(f"Uploading {model}/model.safetensors ({size_gb:.1f} GB) ...")
        api.upload_file(
            path_or_fileobj=weights,
            path_in_repo=f"{model}/model.safetensors",
            repo_id=args.repo_id,
            commit_message=f"Add {model} weights",
        )
        print(f"  done: {model}")

    print(f"\nAll files uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights_dir",
        required=True,
        help="Local directory holding albedo/ and shading/ sub-directories.",
    )
    parser.add_argument(
        "--repo_id", default="davidserra9/albedo-lbm", help="Target Hub repository."
    )
    parser.add_argument(
        "--private", action="store_true", help="Create the repository as private."
    )
    main(parser.parse_args())
