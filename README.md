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
| `pretrained_256.pth` | Pretraining checkpoint | 256 x 256 | TBD |
| `finetuned_impression_512.pth` | Impression text | 512 x 512 | [`configs/finetuned_impression_512.py`](configs/finetuned_impression_512.py) |
| `finetuned_demographic_impression_512.pth` | Impression text + demographic attributes | 512 x 512 | [`configs/finetuned_demographic_impression_512.py`](configs/finetuned_demographic_impression_512.py) |
| `finetuned_control_siim_512.pth` | Control-conditioned generation | 512 x 512 | TBD |

The quick start below demonstrates text-conditioned generation with `finetuned_impression_512.pth`. Other checkpoints should be paired with their matching config and input format.

### Request Weights

1. Obtain [PhysioNet credentialed access](https://physionet.org/settings/credentialing/) by completing the required training.
2. Fill out the [Model Access Request Form](https://docs.google.com/forms/d/e/1FAIpQLSdb9grrTKpslvaaRmShY86nPyv2478fVm9VsELPPkTTzQR6Sg/viewform).
3. After approval, place downloaded checkpoints under `weights/`.

Recommended layout:

```text
weights/
  finetuned_impression_512.pth
  finetuned_demographic_impression_512.pth
  finetuned_control_siim_512.pth
  pretrained_256.pth
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

## Text to Xray Generation

The default text-to-X-ray path uses `weights/finetuned_impression_512.pth` paired with [`configs/finetuned_impression_512.py`](configs/finetuned_impression_512.py). The sections below all use this 512 setup as the running example; swap in another checkpoint + matching config to generate at a different resolution (see [Other Resolutions](#other-resolutions)).

### Quick Start

After placing `weights/finetuned_impression_512.pth`, pick one of the wrappers:

```bash
bash scripts/sample_impression.sh        # five built-in impression prompts
bash scripts/sample_csv.sh    # read prompts from data/mimic_val_p19_impression_example.csv
```

`sample_csv.sh` also accepts a custom CSV path:

```bash
bash scripts/sample_csv.sh path/to/your_prompts.csv
```

Outputs land in `visualization/` along with a `prompts.txt` mapping each image to its source prompt.

### Direct Sampler Invocation

For one-off prompts or non-default flags, call `tools/sample.py` directly:

```bash
torchrun \
    --nproc_per_node=1 \
    --master_port=12345 \
    tools/sample.py configs/finetuned_impression_512.py weights/finetuned_impression_512.pth \
    --work-dir output/ \
    --text-prompt "Moderate cardiomegaly with mild vascular congestion." \
    --cfg-scale 4.0 \
    --num-sampling-steps 100 \
    --seed 1234
```

### Prompt Files

Pass `--text-prompt-file` instead of `--text-prompt` to batch over a file:

```bash
torchrun \
    --nproc_per_node=1 \
    --master_port=12345 \
    tools/sample.py configs/finetuned_impression_512.py weights/finetuned_impression_512.pth \
    --work-dir output/ \
    --text-prompt-file prompts.csv
```

Supported formats:

- `.csv`: use `--text-prompt-key` for the prompt column. A `name` column is optional and controls output file names.
- `.json`: list of strings or list of objects containing the prompt key.
- `.jsonl`: one string or object per line.
- `.txt`: one prompt per line.

A ready-to-run example ships at [`data/mimic_val_p19_impression_example.csv`](data/mimic_val_p19_impression_example.csv) (10 rows; columns `name`, `Finding Labels`, `impression`):

```csv
name,Finding Labels,impression
chest_001.png,No Finding,No acute cardiopulmonary abnormality. ...
chest_002.png,Cardiomegaly,Mild enlargement of the cardiac silhouette without overt pulmonary edema.
chest_003.png,Pleural Effusion,New small loculated pleural effusion within the major fissure of the right lung.
```

The default `--text-prompt-key` is `impression`; override only if your CSV uses a different column name. If the `name` column is omitted, outputs are saved as `0.png`, `1.png`, ... by row index.

### Multi-GPU Sampling

The sampler uses `DistributedSampler` to shard prompts across ranks, so larger prompt files can be parallelised across GPUs. Set `--nproc_per_node` to the number of available GPUs (and update `NUM_GPUS` in `scripts/sample_impression.sh` if you use the wrapper):

```bash
torchrun \
    --nproc_per_node=4 \
    --master_port=12345 \
    tools/sample.py configs/finetuned_impression_512.py weights/finetuned_impression_512.pth \
    --work-dir output/ \
    --text-prompt-file prompts.csv
```

### Common Parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `--cfg-scale` | `4.0` | Classifier-free guidance scale |
| `--num-sampling-steps` | `100` | Number of diffusion denoising steps |
| `--seed` | `0` | Random seed |
| `--batch-size` | `1` | Batch size per GPU. Use `1` for best fidelity; larger values run faster but route cross-attention through `xformers` `BlockDiagonalMask`, whose differing reduction order can introduce small numerical drift across the 100 sampling steps. |
| `--text-prompt-file` | `None` | Prompt file path |
| `--text-prompt-key` | `impression` | Prompt field for CSV/JSON/JSONL files |

## Project Structure

```text
ChexGen/
  configs/            Model configurations
  data/               Example prompt CSV (MIMIC-CXR validation slice)
  radiffuser/
    models/           DiT, T5 text encoder, embedders, control modules
    diffusion/        Gaussian diffusion and timestep respacing
    datasets/         Text and text-to-image dataset loaders
    utils/            Checkpoint loading, logging, synchronization
  scripts/            Shell scripts for generation (sample_impression.sh, sample_csv.sh, sample_demographic_impression.sh)
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
