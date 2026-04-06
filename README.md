# ChexGen: A Generative Foundation Model for Chest Radiography

<p align="center">
  <a href="https://arxiv.org/abs/2509.03903"><img src="https://img.shields.io/badge/arXiv-2509.03903-b31b1b.svg" alt="arXiv"></a>
  <a href="https://ai.nejm.org"><img src="https://img.shields.io/badge/NEJM AI-Accepted-blue.svg" alt="NEJM AI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

## News
- **[2025.09]** Paper released on [arXiv](https://arxiv.org/abs/2509.03903).
- **[2026.03]** ChexGen has been accepted by **NEJM AI**.

## Introduction

**ChexGen** is a generative foundation model for chest radiography that synthesizes realistic chest X-rays conditioned on text prompts, masks, and bounding boxes. Built on a latent diffusion transformer (DiT) architecture and trained on 960,000 radiograph-report pairs, ChexGen can:

- Improve disease **classification**, **detection**, and **segmentation** with less training data
- Generate high-quality **synthetic data** for downstream model training
- Enable **bias detection and mitigation** to enhance fairness across demographic groups

## Getting Started

### Installation

```bash
git clone https://github.com/YuanfengJi/ChexGen.git
cd ChexGen
pip install -r requirements.txt
```

Key dependencies: PyTorch >= 2.2.0, Transformers, Diffusers, xFormers, MMEngine.

### Model Weights

The training data includes [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/), which requires credentialed access via [PhysioNet](https://physionet.org/). We therefore require verification before granting access to model weights.

Currently, we provide the text-conditioned generation weights. Weights for mask- and bounding-box-conditioned generation will be released in the future.

| Model | Condition | Resolution | 
|-------|-----------|-----------|
| ChexGen | Text | 512 x 512 |

**Steps to access:**
1. Obtain [PhysioNet credentialed access](https://physionet.org/settings/credentialing/) by completing the [CITI training course](https://about.citiprogram.org/)
2. Fill out our [Model Access Request Form](https://docs.google.com/forms/d/e/1FAIpQLSdb9grrTKpslvaaRmShY86nPyv2478fVm9VsELPPkTTzQR6Sg/viewform) and upload your CITI certificate
3. We will review your request and reply with the download link via email

After downloading, place the weights under `weights/`:

```bash
mkdir -p weights
# move downloaded file to weights/finetune_impression_512.pth
```

## Usage

### Quick Start

```bash
bash scripts/sample.sh
```

This generates CXR images from predefined radiology text prompts and saves them to `visualization/`. The script includes example prompts covering common findings such as cardiomegaly, pleural effusions, pulmonary masses, and normal chest X-rays.

### Custom Generation

Edit the prompts in `scripts/sample.sh`, or call the generation script directly:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12345 \
    tools/sample.py configs/model.py weights/finetune_impression_512.pth \
    --work-dir output/ \
    --text-prompt "Moderate cardiomegaly with mild vascular congestion." \
    --cfg-scale 4.0 \
    --seed 1234
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cfg-scale` | 4.0 | Classifier-free guidance scale |
| `--num-sampling-steps` | 100 | Number of diffusion denoising steps |
| `--seed` | 0 | Random seed for reproducibility |
| `--batch-size` | 1 | Batch size per GPU |
| `--text-prompt-file` | — | JSON/CSV file with prompts (alternative to inline prompts) |

## Project Structure

```
ChexGen/
├── configs/            # Model configurations
├── radiffuser/
│   ├── models/         # DiT, T5 text encoder, embedders
│   ├── diffusion/      # Gaussian diffusion, timestep sampling
│   ├── datasets/       # Text-to-image dataset loaders
│   └── utils/          # Checkpoint loading, logging
├── scripts/            # Shell scripts for generation
└── tools/              # Entry-point scripts (sample.py)
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{ji2025chexgen,
  title={A Generative Foundation Model for Chest Radiography},
  author={Ji, Yuanfeng and Lin, Dan and Wang, Xiyue and Zhang, Lu and Zhou, Wenhui and Ge, Chongjian and Chu, Ruihang and Yang, Xiaoli and Zhao, Junhan and Chen, Junsong and Luo, Xiangde and Yang, Sen and Fang, Jin and Luo, Ping and Li, Ruijiang},
  journal={NEJM AI},
  year={2025}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).

## Acknowledgements

This codebase builds upon [DiT](https://github.com/facebookresearch/DiT) and [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha). We thank the authors for their excellent work.
