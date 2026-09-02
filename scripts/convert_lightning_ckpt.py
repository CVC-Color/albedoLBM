"""Convert a PyTorch Lightning training checkpoint into inference-ready safetensors.

The training checkpoints produced by our pipeline weigh ~15 GB because they also
carry the AdamW optimizer state, the LPIPS weights used by the training losses and
the pickled training configs. Only the VAE and the denoiser are needed at inference
time (~5 GB in bfloat16), so this script:

  1. reads ``state_dict`` from the Lightning checkpoint,
  2. renames the keys to the ones expected by :class:`albedo_lbm.models.lbm.LBMModel`,
  3. drops the training-only tensors (LPIPS),
  4. writes a single ``model.safetensors`` file.

Usage::

    python scripts/convert_lightning_ckpt.py \
        --checkpoint /path/to/reflectance-best.ckpt \
        --output_dir weights/albedo
"""

import argparse
import io
import os
import pickle
import sys
from typing import Dict

import torch
from safetensors.torch import save_file

# Keys that only matter while training and are not part of the inference graph.
DISCARDED_PREFIXES = ("lpips_loss.",)


class _Placeholder:
    """Stand-in for training classes that are not shipped with this repository."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    @classmethod
    def _reconstruct(cls, *args, **kwargs):
        return cls()


class _ShimUnpickler(pickle.Unpickler):
    """Unpickler that tolerates missing training-time classes.

    Lightning stores the training configs (``TrainingConfig``, ``LBMConfig``, ...)
    inside the checkpoint under ``hyper_parameters``. Those live in modules that are
    not part of this inference-only release, so we replace anything we cannot import
    with a placeholder instead of failing. Tensors are unaffected: they are restored
    by ``torch`` through its own persistent-load hook.
    """

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _Placeholder


class _shim_pickle_module:
    Unpickler = _ShimUnpickler
    load = pickle.load
    dump = pickle.dump
    Pickler = pickle.Pickler
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def remap_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map Lightning checkpoint keys onto ``LBMModel`` attribute names."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        key = key.replace("vae.vae_encoder", "vae.vae_model.encoder")
        key = key.replace("vae.vae_decoder", "vae.vae_model.decoder")
        key = key.replace("vae.vae_quant_conv", "vae.vae_model.quant_conv")
        key = key.replace("vae.vae_post_quant_conv", "vae.vae_model.post_quant_conv")
        if key.startswith(DISCARDED_PREFIXES):
            continue
        remapped[key] = value.contiguous()
    return remapped


def main(args: argparse.Namespace) -> None:
    print(f"Reading {args.checkpoint} ...")
    raw = torch.load(
        args.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
        pickle_module=_shim_pickle_module,
    )

    state_dict = raw["state_dict"] if "state_dict" in raw else raw
    print(f"  {len(state_dict)} tensors in the checkpoint")

    state_dict = remap_keys(state_dict)
    if args.dtype != "keep":
        target_dtype = getattr(torch, args.dtype)
        state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}

    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"  {len(state_dict)} tensors kept ({total_bytes / 1e9:.2f} GB)")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "model.safetensors")
    save_file(state_dict, output_path, metadata={"format": "pt"})
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the Lightning .ckpt file."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where model.safetensors will be written.",
    )
    parser.add_argument(
        "--dtype",
        default="keep",
        choices=["keep", "bfloat16", "float16", "float32"],
        help="Cast the weights before saving (default: keep the checkpoint dtype).",
    )
    main(parser.parse_args())
