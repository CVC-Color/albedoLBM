"""Estimate albedo (and shading) for one image or a folder of images.

Examples::

    # Full two-stage pipeline, weights downloaded from the Hugging Face Hub
    python scripts/infer.py --input assets/examples --output results

    # Condition on a shading image you already have (single model, less memory)
    python scripts/infer.py --input image.png --shading shading.png --output results

    # Local weights, 4 bridge steps instead of 1
    python scripts/infer.py --input image.png --output results \
        --model_dir weights --num_steps 4
"""

import argparse
import logging
import os
import time
from glob import glob

import torch
from PIL import Image

from albedo_lbm import IntrinsicDecomposer
from albedo_lbm.pipeline import DEFAULT_REPO

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

logger = logging.getLogger("albedo_lbm")


def collect_images(path: str):
    """Return the list of images to process, from a file or a directory."""
    if os.path.isdir(path):
        files = sorted(
            f for f in glob(os.path.join(path, "*")) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not files:
            raise FileNotFoundError(f"No image found in {path}")
        return files
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return [path]


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    images = collect_images(args.input)
    logger.info("Found %d image(s)", len(images))

    pipeline = IntrinsicDecomposer.from_pretrained(
        args.model_dir,
        load_shading_model=args.shading is None,
        device=args.device,
        torch_dtype=getattr(torch, args.dtype),
    )

    shading = Image.open(args.shading).convert("RGB") if args.shading else None
    os.makedirs(args.output, exist_ok=True)

    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        image = Image.open(path).convert("RGB")

        start = time.time()
        result = pipeline(image, shading=shading, num_steps=args.num_steps)
        elapsed = time.time() - start

        result.albedo.save(os.path.join(args.output, f"{name}_albedo.png"))
        if args.save_shading and result.shading is not None:
            result.shading.save(os.path.join(args.output, f"{name}_shading.png"))
        if args.save_reconstruction and result.shading is not None:
            result.reconstruction().save(
                os.path.join(args.output, f"{name}_reconstruction.png")
            )

        logger.info("%s (%dx%d) in %.2fs", name, *image.size, elapsed)

    logger.info("Results written to %s", os.path.abspath(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, help="Input image, or a directory of images."
    )
    parser.add_argument(
        "--output", default="results", help="Directory for the predictions."
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_REPO,
        help="Hugging Face repository id, or a local directory holding the "
        "albedo/ and shading/ sub-directories.",
    )
    parser.add_argument(
        "--shading",
        default=None,
        help="Optional shading image used as conditioning. When given, the shading "
        "model is not loaded and this image is used instead.",
    )
    parser.add_argument(
        "--num_steps", type=int, default=1, help="Bridge steps per model (default: 1)."
    )
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Inference precision (default: bfloat16).",
    )
    parser.add_argument(
        "--save_shading", action="store_true", help="Also save the estimated shading."
    )
    parser.add_argument(
        "--save_reconstruction",
        action="store_true",
        help="Also save albedo * shading, to inspect the image formation model.",
    )
    main(parser.parse_args())
