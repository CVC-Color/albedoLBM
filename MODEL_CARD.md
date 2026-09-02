---
license: cc-by-nc-4.0
tags:
  - intrinsic-image-decomposition
  - albedo
  - shading
  - latent-bridge-matching
  - image-to-image
library_name: albedo-lbm
pipeline_tag: image-to-image
---

# Albedo Estimation via Latent Bridge Matching (CIC 2026)

Weights for **Albedo Estimation via Latent Bridge Matching**, by Carme Corbi,
David Serrano-Lozano, Javier Vazquez-Corral and Maria Vanrell (Universitat
Autònoma de Barcelona and Computer Vision Center), published at the Color and
Imaging Conference 2026.

Code: [github.com/CVC-Color/albedoLBM](https://github.com/CVC-Color/albedoLBM)

## Contents

| Sub-folder | Model | Task | Conditioning |
| --- | --- | --- | --- |
| `albedo/` | LBM-AID | RGB → albedo | shading |
| `shading/` | LBM-SID | RGB → shading | albedo |

Each sub-folder holds a `config.yaml` and a `model.safetensors` (5.0 GB,
bfloat16): a Stable Diffusion XL UNet (~2.5B parameters) used as the drift
network, plus the frozen SDXL VAE. The two models are used together as a
two-stage decomposition pipeline.

## Usage

```python
from PIL import Image
from albedo_lbm import IntrinsicDecomposer

pipeline = IntrinsicDecomposer.from_pretrained("davidserra9/albedo-lbm")
result = pipeline(Image.open("image.png"))

result.albedo.save("albedo.png")
result.shading.save("shading.png")
```

Install the code with `pip install -e .` from the
[repository](https://github.com/CVC-Color/albedoLBM).

## Training

Both models were trained for the paper on InteriorVerse (~44K images) and
Hypersim (~59K images), on 256×256 random crops, with AdamW at a learning rate of
4e-5. The drift network is initialized from Stable Diffusion XL; the VAE stays
frozen. The albedo model is trained with the pixel-space reconstruction loss that
enforces the image formation model `I = A · S`; the shading model is trained
without it, which we found gives better shading estimates.

Sampling uses four equally spaced timesteps at training time and, by default, a
single bridge step at inference.

## Limitations

- Trained on synthetic indoor data only, so real-world images (especially
  outdoor scenes, people and objects) can show color shifts.
- Transparent, metallic and strongly non-Lambertian surfaces remain a failure
  case.
- Quality degrades above roughly 2K resolution.

## Citation

```bibtex
@inproceedings{corbi2026albedo,
  title     = {Albedo Estimation via Latent Bridge Matching},
  author    = {Corbi, Carme and Serrano-Lozano, David and Vazquez-Corral, Javier and Vanrell, Maria},
  booktitle = {Color and Imaging Conference (CIC)},
  year      = {2026}
}
```

## License

CC BY-NC 4.0, following the license of the upstream
[LBM](https://github.com/gojasper/LBM) code these models build on. Use is also
subject to the terms of the InteriorVerse and Hypersim datasets and of the
Stable Diffusion XL backbone.
