import logging
import os
from typing import List, Optional

import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from albedo_lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from albedo_lbm.models.lbm import LBMConfig, LBMModel
from albedo_lbm.models.unets import DiffusersUNet2DCondWrapper
from albedo_lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.yaml"
WEIGHTS_NAME = "model.safetensors"


def get_model(
    model_dir: str,
    subfolder: Optional[str] = None,
    cache_dir: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> LBMModel:
    """Build an LBM model and load its weights.

    Args:
        model_dir (str): Either a local directory holding ``config.yaml`` and
            ``model.safetensors``, or the id of a Hugging Face Hub repository
            (e.g. ``"davidserra9/albedo-lbm"``).
        subfolder (Optional[str]): Sub-directory inside ``model_dir`` holding the
            model files. Use it to pick one of the models of a multi-model
            repository (e.g. ``"albedo"`` or ``"shading"``).
        cache_dir (Optional[str]): Where to store the files downloaded from the Hub.
        torch_dtype (torch.dtype): dtype the model is cast to.
        device (str): Device the model is moved to.

    Returns:
        LBMModel: the model, in eval mode, ready for inference.
    """
    config_path, weights_path = _resolve_files(model_dir, subfolder, cache_dir)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config: %s", config_path)

    model = _get_model_from_config(**config, torch_dtype=torch_dtype)

    logger.info("Loading weights: %s", weights_path)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.to(torch_dtype)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def _resolve_files(model_dir: str, subfolder: Optional[str], cache_dir: Optional[str]):
    """Return the local paths of the config and the weights, downloading if needed."""
    local_dir = os.path.join(model_dir, subfolder) if subfolder else model_dir

    if os.path.isdir(local_dir):
        config_path = os.path.join(local_dir, CONFIG_NAME)
        weights_path = os.path.join(local_dir, WEIGHTS_NAME)
        for path in (config_path, weights_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path} in model directory {local_dir}")
        return config_path, weights_path

    if os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"{model_dir} is a local directory but has no {subfolder}/ sub-directory. "
            "A local model directory must hold one sub-directory per model, each with "
            f"a {CONFIG_NAME} and a {WEIGHTS_NAME}."
        )

    # Not a local directory: treat it as a Hugging Face Hub repository id.
    download = lambda filename: hf_hub_download(  # noqa: E731
        repo_id=model_dir,
        filename=filename,
        subfolder=subfolder,
        cache_dir=cache_dir,
    )
    return download(CONFIG_NAME), download(WEIGHTS_NAME)


def _get_model_from_config(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
    source_key: str = "source_image",
    target_key: str = "target_image",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    torch_dtype: torch.dtype = torch.bfloat16,
    transformer_layers_per_block: Optional[List[int]] = None,
    **kwargs,
) -> LBMModel:
    conditioning_images_keys = conditioning_images_keys or []
    conditioning_masks_keys = conditioning_masks_keys or []

    ## Denoiser ##
    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # latent + conditioning latent
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=transformer_layers_per_block,
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch_dtype)

    ## Conditioner ##
    conditioners = []
    if conditioning_images_keys or conditioning_masks_keys:
        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    conditioner = ConditionerWrapper(conditioners=conditioners)

    ## VAE ##
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config).to(torch_dtype)
    vae.freeze()
    vae.to(torch_dtype)

    ## Latent bridge ##
    config = LBMConfig(
        source_key=source_key,
        target_key=target_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    model = LBMModel(
        config,
        denoiser=denoiser,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch_dtype)

    return model
