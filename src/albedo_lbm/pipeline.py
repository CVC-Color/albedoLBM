"""Two-stage intrinsic decomposition pipeline.

Albedo and shading are mutually informative: the albedo model is conditioned on a
shading estimate, and the shading model is conditioned on an albedo estimate. The
pipeline resolves this circular dependency with a bootstrap pass, as described in
the paper (Experiment 3):

    1. bootstrap  : albedo_0 = LBM-AID(image, conditioning = bootstrap image)
    2. shading    : shading = LBM-SID(image, conditioning = albedo_0)
    3. albedo     : albedo  = LBM-AID(image, conditioning = shading)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import PIL
import torch
from PIL import Image

from .inference import evaluate, get_model
from .models.lbm import LBMModel

logger = logging.getLogger(__name__)

DEFAULT_REPO = "davidserra9/albedo-lbm"
ALBEDO_SUBFOLDER = "albedo"
SHADING_SUBFOLDER = "shading"


@dataclass
class IntrinsicDecomposition:
    """Result of a decomposition. All images share the size of the input."""

    albedo: PIL.Image.Image
    shading: Optional[PIL.Image.Image] = None
    bootstrap_albedo: Optional[PIL.Image.Image] = None

    def reconstruction(self) -> PIL.Image.Image:
        """Re-synthesize the input as ``albedo * shading`` (image formation model)."""
        if self.shading is None:
            raise ValueError("A shading estimate is needed to reconstruct the input.")
        from torchvision.transforms import ToPILImage, ToTensor

        albedo = ToTensor()(self.albedo)
        shading = ToTensor()(self.shading)
        return ToPILImage()((albedo * shading).clamp(0, 1))


class IntrinsicDecomposer:
    """Runs the albedo (and optionally the shading) LBM models on an image.

    Args:
        albedo_model (LBMModel): The shading-conditioned albedo model (LBM-AID).
        shading_model (Optional[LBMModel]): The albedo-conditioned shading model
            (LBM-SID). When omitted, a shading image must be supplied at call time.
    """

    def __init__(
        self, albedo_model: LBMModel, shading_model: Optional[LBMModel] = None
    ):
        self.albedo_model = albedo_model
        self.shading_model = shading_model

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str = DEFAULT_REPO,
        load_shading_model: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ) -> "IntrinsicDecomposer":
        """Load the pipeline from a Hugging Face repository or a local directory.

        Args:
            model_dir (str): Hub repository id or local directory holding the
                ``albedo/`` and ``shading/`` sub-directories.
            load_shading_model (bool): Load the shading model as well. Set to False
                to save memory when you already have a shading image.
            device (str): Device the models are moved to.
            torch_dtype (torch.dtype): dtype the models are cast to.
            cache_dir (Optional[str]): Where to store files downloaded from the Hub.
        """
        albedo_model = get_model(
            model_dir,
            subfolder=ALBEDO_SUBFOLDER,
            device=device,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        shading_model = None
        if load_shading_model:
            shading_model = get_model(
                model_dir,
                subfolder=SHADING_SUBFOLDER,
                device=device,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
            )
        return cls(albedo_model=albedo_model, shading_model=shading_model)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, str],
        shading: Optional[PIL.Image.Image] = None,
        num_steps: int = 1,
        return_intermediate: bool = False,
    ) -> IntrinsicDecomposition:
        """Decompose an image into albedo and shading.

        Args:
            image (PIL.Image.Image or str): Input RGB image, or a path to one.
            shading (Optional[PIL.Image.Image]): Use this shading image to condition
                the albedo model instead of estimating one. Required when the
                pipeline was loaded without the shading model.
            num_steps (int): Number of bridge steps per model. Defaults to 1.
            return_intermediate (bool): Also return the bootstrap albedo.

        Returns:
            IntrinsicDecomposition: the estimated albedo and shading.
        """
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        bootstrap_albedo = None

        if shading is None:
            if self.shading_model is None:
                raise ValueError(
                    "This pipeline was loaded without the shading model, so a "
                    "`shading` image must be given."
                )
            # 1. Bootstrap albedo: the shading model needs an albedo estimate, and
            #    the albedo model needs a shading estimate. We break the loop by
            #    conditioning the first albedo pass on the input image itself.
            bootstrap_albedo = evaluate(
                self.albedo_model, image, image, num_sampling_steps=num_steps
            )
            # 2. Shading, conditioned on that first albedo estimate.
            shading = evaluate(
                self.shading_model,
                image,
                bootstrap_albedo,
                num_sampling_steps=num_steps,
            )
        else:
            shading = shading.convert("RGB")
            if shading.size != image.size:
                shading = shading.resize(image.size)

        # 3. Final albedo, conditioned on the shading estimate.
        albedo = evaluate(
            self.albedo_model, image, shading, num_sampling_steps=num_steps
        )

        return IntrinsicDecomposition(
            albedo=albedo,
            shading=shading,
            bootstrap_albedo=bootstrap_albedo if return_intermediate else None,
        )
