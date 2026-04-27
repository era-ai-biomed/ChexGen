# ChexGen: A Generative Foundation Model for Chest Radiography

<p align="center">
  <a href="https://arxiv.org/abs/2509.03903"><img src="https://img.shields.io/badge/arXiv-2509.03903-b31b1b.svg" alt="arXiv"></a>
  <a href="https://ai.nejm.org"><img src="https://img.shields.io/badge/NEJM%20AI-Accepted-blue.svg" alt="NEJM AI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

ChexGen is a generative foundation model for chest radiography. It synthesizes chest X-rays from radiology text prompts and optional control inputs, and is designed to support synthetic data generation, downstream model development, and research on robustness and fairness in medical imaging.

## News

- **[2026.03]** ChexGen has been accepted by **NEJM AI**. [Published article](paper/AIoa2500799.pdf) and [supplementary appendix](paper/aioa2500799_appendix.pdf) are available.
- **[2025.09]** Paper released on [arXiv](https://arxiv.org/abs/2509.03903).

## Highlights

- Latent diffusion transformer (DiT) architecture for chest radiograph generation.
- Trained on 960,000 radiograph-report pairs.
- Text- and control-conditioned generation from radiology impressions.
- Checkpoints for multiple resolutions and conditioning settings.
- Open research code for model loading, text embedding, diffusion sampling, and dataset utilities.

## Available Checkpoints

The training data includes [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/), which requires credentialed access via [PhysioNet](https://physionet.org/). For that reason, access to model weights requires verification.

| Checkpoint | Condition | Resolution | Config |
| --- | --- | ---: | --- |
| `pretrain_256.pth` | Pretraining checkpoint | 256 x 256 | TBD |
| `finetune_impression_512.pth` | Impression text | 512 x 512 | [`configs/model.py`](configs/model.py) |
| `finetune_impression_1024.pth` | Impression text | 1024 x 1024 | TBD |
| `finetune_impression_sex_age_race_512.pth` | Impression text + demographic attributes | 512 x 512 | TBD |
| `finetune_control_siim_512.pth` | Control-conditioned generation | 512 x 512 | TBD |

The quick start below demonstrates text-conditioned generation with `finetune_impression_512.pth`. Other checkpoints should be paired with their matching config and input format.

### Request Weights

1. Obtain [PhysioNet credentialed access](https://physionet.org/settings/credentialing/) by completing the required training.
2. Fill out the [Model Access Request Form](https://docs.google.com/forms/d/e/1FAIpQLSdb9grrTKpslvaaRmShY86nPyv2478fVm9VsELPPkTTzQR6Sg/viewform).
3. After approval, place downloaded checkpoints under `weights/`.

Recommended layout:

```text
weights/
  finetune_impression_512.pth
  finetune_impression_1024.pth
  finetune_impression_sex_age_race_512.pth
  finetune_control_siim_512.pth
  pretrain_256.pth
```

Checkpoint files are intentionally ignored by git.

## Installation

```bash
git clone https://github.com/era-ai-biomed/ChexGen.git
cd ChexGen
pip install -r requirements.txt
pip install -e .
```

Key dependencies include PyTorch, torchvision, Transformers, Diffusers, xFormers, MMEngine, timm, and sentencepiece. The sampling script expects an NVIDIA GPU with CUDA and runs through PyTorch distributed sampling with the NCCL backend.

The first run may download additional public model components:

- `DeepFloyd/t5-v1_1-xxl` for text embeddings.
- `stabilityai/sd-vae-ft-ema` or `stabilityai/sd-vae-ft-mse` for latent decoding.

If you are running on a restricted cluster, pre-download these assets into the Hugging Face cache before launching sampling.

## Quick Start

After placing `weights/finetune_impression_512.pth`, run:

```bash
bash scripts/sample.sh
```

Generated images are saved to `visualization/`.

## Custom Text Generation

You can edit the prompt list in `scripts/sample.sh`, or call the sampler directly:

```bash
torchrun \
    --nproc_per_node=1 \
    --master_port=12345 \
    tools/sample.py configs/model.py weights/finetune_impression_512.pth \
    --work-dir output/ \
    --text-prompt "Moderate cardiomegaly with mild vascular congestion." \
    --cfg-scale 4.0 \
    --num-sampling-steps 100 \
    --seed 1234
```

Prompt files are also supported:

```bash
torchrun \
    --nproc_per_node=1 \
    --master_port=12345 \
    tools/sample.py configs/model.py weights/finetune_impression_512.pth \
    --work-dir output/ \
    --text-prompt-file prompts.csv \
    --text-prompt-key caption
```

Supported prompt file formats:

- `.csv`: use `--text-prompt-key` for the prompt column. A `name` column is optional and controls output file names.
- `.json`: list of strings or list of objects containing the prompt key.
- `.jsonl`: one string or object per line.
- `.txt`: one prompt per line.

### Common Parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `--cfg-scale` | `4.0` | Classifier-free guidance scale |
| `--num-sampling-steps` | `100` | Number of diffusion denoising steps |
| `--seed` | `0` | Random seed |
| `--batch-size` | `1` | Batch size per GPU |
| `--text-prompt-file` | `None` | Prompt file path |
| `--text-prompt-key` | `caption` | Prompt field for CSV/JSON/JSONL files |

For resolutions other than 512 x 512, use a matching config. The latent `input_size` should be 32 for 256 x 256 images, 64 for 512 x 512 images, and 128 for 1024 x 1024 images.

## Project Structure

```text
ChexGen/
  configs/            Model configurations
  radiffuser/
    models/           DiT, T5 text encoder, embedders, control modules
    diffusion/        Gaussian diffusion and timestep respacing
    datasets/         Text and text-to-image dataset loaders
    utils/            Checkpoint loading, logging, synchronization
  scripts/            Shell scripts for generation
  tools/              Entry-point scripts
```

## Intended Use and Limitations

ChexGen is released for research use. Generated images are synthetic and should not be used as a substitute for clinical imaging, diagnosis, treatment planning, or medical decision-making. Users are responsible for validating synthetic data in their own downstream settings, including checks for artifacts, label leakage, demographic bias, and task-specific failure modes.

## Citation

If you find this work useful, please cite:

```bibtex
@article{ji2026generative,
  title={A generative foundation model for chest radiography},
  author={Ji, Yuanfeng and Lin, Dan and Wang, Xiyue and Zhang, Lu and Zhou, Wenhui and Ge, Chongjian and Chu, Ruihang and Yang, Xiaoli and Zhao, Junhan and Chen, Junsong and others},
  journal={NEJM AI},
  pages={AIoa2500799},
  year={2026},
  publisher={Massachusetts Medical Society}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).

## Acknowledgements

This codebase builds upon [DiT](https://github.com/facebookresearch/DiT) and [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha). We thank the authors for their excellent work.
