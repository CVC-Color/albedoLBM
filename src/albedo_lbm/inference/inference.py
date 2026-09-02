import logging
from typing import Optional, Union

import PIL
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from albedo_lbm.models.lbm import LBMModel

logger = logging.getLogger(__name__)

ImageLike = Union[PIL.Image.Image, torch.Tensor]


def to_model_input(
    image: ImageLike, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Turn a PIL image or a ``[C, H, W]`` tensor into a ``[1, C, H, W]`` batch in [-1, 1].

    Tensors already in [-1, 1] are passed through unchanged; tensors in [0, 1]
    (and PIL images) are rescaled, since the VAE of SDXL expects [-1, 1] inputs.
    """
    if isinstance(image, Image.Image):
        tensor = ToTensor()(image).unsqueeze(0) * 2 - 1

    elif isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise ValueError(
                f"Image tensors must have shape [C, H, W], got {tuple(image.shape)}."
            )
        if image.max() <= 1.0 and image.min() >= 0.0:
            image = image * 2 - 1
        tensor = image.unsqueeze(0)

    else:
        raise TypeError(
            f"Expected a PIL.Image.Image or a torch.Tensor, got {type(image)}."
        )

    return tensor.to(device=device, dtype=dtype)


def to_pil(tensor: torch.Tensor, size: Optional[tuple] = None) -> PIL.Image.Image:
    """Convert a ``[1, C, H, W]`` model output in [-1, 1] back to a PIL image."""
    image = (tensor[0].float().cpu() + 1) / 2
    image = ToPILImage()(image.clamp(0, 1))
    if size is not None:
        image = image.resize(size)
    return image


@torch.no_grad()
def evaluate(
    model: LBMModel,
    source_image: ImageLike,
    conditioning_image: Optional[ImageLike] = None,
    num_sampling_steps: int = 1,
) -> ImageLike:
    """Run a single forward pass of an LBM model.

    Args:
        model (LBMModel): The LBM model used for inference.
        source_image (PIL.Image.Image or torch.Tensor): Input RGB image. Tensors must
            have shape ``[C, H, W]`` and be normalized to either [0, 1] or [-1, 1].
        conditioning_image (PIL.Image.Image or torch.Tensor, optional): Conditioning
            image (shading for the albedo model, albedo for the shading model). It
            must have the same spatial dimensions as ``source_image``.
        num_sampling_steps (int): Number of bridge steps. Defaults to 1.

    Returns:
        A PIL image if ``source_image`` is a PIL image, a tensor otherwise.
    """
    device, dtype = model.device, model.dtype

    batch = {model.source_key: to_model_input(source_image, device, dtype)}

    if conditioning_image is not None:
        conditioning_keys = _conditioning_keys(model)
        if not conditioning_keys:
            logger.warning(
                "A conditioning image was given but this model is unconditional; "
                "it will be ignored."
            )
        conditioning_tensor = to_model_input(conditioning_image, device, dtype)
        if conditioning_tensor.shape[-2:] != batch[model.source_key].shape[-2:]:
            raise ValueError(
                "The conditioning image must have the same spatial size as the source "
                f"image, got {tuple(conditioning_tensor.shape[-2:])} and "
                f"{tuple(batch[model.source_key].shape[-2:])}."
            )
        for key in conditioning_keys:
            batch[key] = conditioning_tensor

    elif _conditioning_keys(model):
        raise ValueError(
            "This model expects a conditioning image "
            f"({', '.join(_conditioning_keys(model))}), but none was given."
        )

    # The bridge starts from the source image itself rather than from Gaussian noise.
    z_source = model.vae.encode(batch[model.source_key])

    output = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    if isinstance(source_image, Image.Image):
        return to_pil(output, size=source_image.size)

    return output


def _conditioning_keys(model: LBMModel):
    """List the batch keys the conditioners of ``model`` read their images from."""
    keys = []
    for conditioner in model.conditioner.conditioners:
        keys.extend(getattr(conditioner.config, "image_keys", []))
    return keys
