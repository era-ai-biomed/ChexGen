# ChexGen — MIMIC-CXR data-augmentation reproduction kit

Exact pipeline used in the ChexGen paper for the MIMIC-CXR multi-label classification
**data-augmentation** experiment (real images vs. real + ChexGen-generated synthetic).
Built on top of [MedFM](https://github.com/openmedlab/MedFM) (mmcls 0.25.0).

## What's in here

```
classification/
├── medfmc/                   # mmcls custom package (MIMIC dataset, AUC_multilabel hook)
├── tools/                    # mmcls train/test entry points
├── configs/
│   ├── _base_/               # backbone, schedule, runtime, dataset configs
│   └── densenet/             # DenseNet121 + MIMIC entrypoints (baseline + sync1..5)
├── data_meta/                # train / val ann files (label files only; no images)
└── work_dirs/mimic_cls/      # our 18 training runs (baseline + sync1..5, 3 seeds each)
    └── <exp>/<seed>/         # *.log, *.log.json, frozen config snapshot
```

## Environment

Trained on PyTorch 2.2 + CUDA 12.1 with:

- `mmcls==0.25.0`
- `mmcv-full==1.6.0`
- `scikit-learn` (for AUC; computed in `medfmc/core/evaluation/eval_metrics.py`)
- 1× A100 80 GB per run (15 epochs, ~30 min for baseline)

```bash
pip install torch torchvision
pip install mmcls==0.25.0 openmim scipy scikit-learn ftfy regex tqdm
mim install mmcv-full==1.6.0
```

## Data preparation

### MIMIC-CXR images

The recipient already needs MIMIC-CXR access. We use the
[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) p10–p18 split for training
and p19 for validation. Edit the two roots in
`configs/_base_/datasets/mimic_256.py` (and the five `mimic_sync<N>_256.py`):

```python
MIMIC_TRAIN_ROOT = 'data/mimic-cxr-jpg/p10_p18'   # contains train/p10*/s*/*.jpg
MIMIC_VAL_ROOT   = 'data/mimic-cxr-jpg/p19'       # contains train/p19*/s*/*.jpg
```

`<root>/train/p<id>/s<id>/view*.jpg` should resolve for every line in the ann files.

### Labels (already in `data_meta/`)

| File | Lines | Contents |
| --- | --- | --- |
| `data_meta/mimic_p10_18_findings_train.txt` | 45 000 | training labels |
| `data_meta/mimic_p19_findings_val.txt` | 3 500 | validation labels |

Format: `<rel_jpg_path> <14 comma-separated 0/1 labels>`, with the 14 columns in
CheXpert order:

```
no finding, enlarged cardiomediastinum, cardiomegaly, airspace opacity,
lung lesion, edema, consolidation, pneumonia, atelectasis, pneumothorax,
pleural effusion, pleural other, fracture, support devices
```

### Backbone

Download the mmcls ImageNet DenseNet121 weight to `weights/`:

```bash
mkdir -p weights
wget -O weights/densenet121_4xb256_in1k_20220426-07450f99.pth \
  https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth
```

### Synthetic data (sync1..5 only)

We ran ChexGen `finetuned_impression_512` (cfg-scale = 4) over the 45 000 MIMIC p10–p18
captions, producing one synthetic image per real caption. The synthetic filenames
match the real ones, so the same ann file is reused — `sync<N>` is built by listing
the same ann file `N+1` times and pairing with `[MIMIC_TRAIN_ROOT, SYN_ROOT, SYN_ROOT, ...]`
(N copies of `SYN_ROOT`), so synthetic is oversampled at N:1 vs. real.

The 45 k synthetic image files are **available on request from the authors**. You can
also regenerate them yourself using ChexGen's `tools/sample.py` against the same
caption CSV (results will be different per-image but distribution-equivalent).

Once you have them, set `SYN_ROOT` in each `configs/_base_/datasets/mimic_sync<N>_256.py`.

## How to run

3-seed sweep per family, 18 runs total:

```bash
cd ChexGen/downstream/classification
export PYTHONPATH=$PWD:$PYTHONPATH

# baseline (real only)
for SEED in 0 1 2; do
  python tools/train.py configs/densenet/dense121_mimic_256.py \
      --work-dir work_dirs/mimic_cls/baseline/$SEED --seed $SEED
done

# sync<N> (real + N× synthetic), N = 1..5
for N in 1 2 3 4 5; do
  for SEED in 0 1 2; do
    python tools/train.py configs/densenet/dense121_mimic_sync${N}_256.py \
        --work-dir work_dirs/mimic_cls/sync${N}/$SEED --seed $SEED
  done
done
```

## Reading our logs

Each `work_dirs/mimic_cls/<exp>/<seed>/*.log.json` is one JSON object per line.
The `mode == "val"` rows give per-epoch overall multi-label AUC (`AUC_multilabel`,
macro-averaged across the 14 classes). The reported per-run metric is the maximum
over the 15 training epochs:

```python
import json, glob, statistics as st
def best(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return max(r['AUC_multilabel'] for r in rows
               if r.get('mode') == 'val' and 'AUC_multilabel' in r)

for fam in ['baseline', 'sync1', 'sync2', 'sync3', 'sync4', 'sync5']:
    seeds = [best(sorted(glob.glob(f'work_dirs/mimic_cls/{fam}/{s}/*.log.json'))[-1])
             for s in '012']
    print(fam, [f'{x:.2f}' for x in seeds],
          f'mean={st.mean(seeds):.2f} std={st.stdev(seeds):.2f}')
```

A subset of seeds has more than one `*.log` / `*.log.json` pair — those are runs that
were resumed after a node failure; `sorted(...)[-1]` picks the final, completed one.

## Reference numbers (overall AUC on p19 val, 3-seed mean ± std)

| Train data | Overall AUC |
| :--- | ---: |
| baseline (45 k real) | 78.02 ± 0.20 |
| real + 1× synth (sync1) | 79.62 ± 0.26 |
| real + 2× synth (sync2) | 80.28 ± 0.18 |
| real + 3× synth (sync3) | 80.59 ± 0.39 |
| real + 4× synth (sync4) | 80.51 ± 0.05 |
| real + 5× synth (sync5) | 80.78 ± 0.16 |

Per-seed numbers are inside the corresponding `work_dirs/mimic_cls/<exp>/<seed>/*.log.json`.

## Notes

- We pick the best-AUC checkpoint by mmcls's `save_best='auto'` hook on the val split.
  Because train/val/test in the configs all point at the same p19 ann file, this is a
  best-on-val number — there is no held-out test split in this experiment.
- Backbone, optimizer (SGD lr=1e-3, cosine), batch size (128), augmentation
  (`RandomResizedCrop` 256 + `RandomFlip`), and number of epochs (15) are inherited
  from the MedFM ChestDR recipe. Changing any of these will move the baseline.
- `work_dirs/mimic_cls/<exp>/<seed>/dense121_mimic_*.py` is the auto-dumped frozen
  config from the original run (paths point at our internal cluster) — kept verbatim
  as the run record. The clean configs you should edit are in `configs/`.
