# Video Sentiment Classification Pipeline

This repository now contains a full video classification pipeline for:

- Binary (`normal` vs `violent`)
- Multiclass (specific violence type)
- Multilabel (e.g., `violent|robbery`)

It supports:

- MP4 decoding and frame sampling
- Per-epoch checkpointing
- Resume after timeout/disconnect
- Validation/test metrics with per-class support
- Local smoke tests with synthetic labels when only one class is available
- Colab auto-resume launcher
- Optional MMAction2 model backend

## Project Files

- `notebooks/Video_Sentiment_Colab_End_to_End.ipynb`: ready-to-run Colab/Jupyter notebook with setup, plotting, training, resume, and evaluation cells.
- `train_video_classifier.py`: training entrypoint (supports `--resume`, `--manifest`, `--output-dir`).
- `test_video_classifier.py`: evaluation entrypoint (supports `--manifest` override).
- `video_pipeline/`: dataset, model, trainer, metrics modules.
- `scripts/extract_ucfcrime_taxonomy.py`: extract label taxonomy from `UCFCrime_Filtered_WithFilename.json`.
- `scripts/build_bootstrap_manifests.py`: build binary/multiclass/multilabel manifests from current normal-only data + synthetic violent labels for testing.
- `scripts/build_video_manifest.py`: build manifest from real class folders.
- `scripts/colab_autoresume.py`: Colab launcher with automatic resume from `checkpoint_last.pt`.
- `scripts/compare_runs.py`: rank and compare run metrics.
- `scripts/colab_setup.sh`: dependency setup helper for Colab.
- `Violent_Videos_Placeholder/`: placeholder folder where all available violent videos can be dropped as one binary class (`violent`).

## 0) Notebook-First Workflow (Recommended)

Open and run:

- `notebooks/Video_Sentiment_Colab_End_to_End.ipynb`

The notebook is structured to run cell-by-cell and includes:

- Colab drive mounting and environment setup.
- Taxonomy extraction from `UCFCrime_Filtered_WithFilename.json`.
- Bootstrap manifest generation from normal-only data.
- Binary manifest generation from one violent folder placeholder.
- Distribution plots.
- Smoke training.
- Auto-resume training launch on Colab.
- Evaluation and per-class F1 plotting.

## 1) Extract Class Taxonomy From Your JSON

Run:

```bash
python scripts/extract_ucfcrime_taxonomy.py \
  --input-json UCFCrime_Filtered_WithFilename.json \
  --output-json manifests/ucfcrime_taxonomy.json
```

With your current file, extracted multiclass labels are:

`normal, abuse, arrest, arson, assault, burglary, explosion, fighting, road_accidents, robbery, shooting, shoplifting, stealing, vandalism`

Multilabel labels are:

`normal, violent, abuse, arrest, arson, assault, burglary, explosion, fighting, road_accidents, robbery, shooting, shoplifting, stealing, vandalism`

## 2) Build Manifests With One-Class Data (Now)

Since you currently only have normal videos, use synthetic violent rows to validate code paths:

```bash
python scripts/build_bootstrap_manifests.py \
  --normal-dir Normal_Videos_for_Event_Recognition/Normal_Videos_for_Event_Recognition \
  --taxonomy-json manifests/ucfcrime_taxonomy.json \
  --output-dir manifests \
  --max-normal 24 \
  --synthetic-train-per-class 1 \
  --synthetic-val-per-class 1 \
  --synthetic-test-per-class 1 \
  --infer-duration
```

Generated files:

- `manifests/bootstrap_binary_manifest.csv`
- `manifests/bootstrap_multiclass_manifest.csv`
- `manifests/bootstrap_multilabel_manifest.csv`
- `manifests/bootstrap_manifest_summary.json`

Important: synthetic rows reuse normal videos with fake labels and are only for pipeline debugging.

## 3) Quick Smoke Training (Local)

Binary:

```bash
python train_video_classifier.py --config configs/smoke_bootstrap_binary.yaml
```

Multiclass (taxonomy-driven class list):

```bash
python train_video_classifier.py --config configs/smoke_bootstrap_multiclass_taxonomy.yaml
```

Multilabel (taxonomy-driven class list):

```bash
python train_video_classifier.py --config configs/smoke_bootstrap_multilabel_taxonomy.yaml
```

Evaluate a checkpoint:

```bash
python test_video_classifier.py \
  --config configs/smoke_bootstrap_multiclass_taxonomy.yaml \
  --checkpoint runs/smoke_bootstrap_multiclass/checkpoint_best.pt \
  --split test
```

## 4) Full Data Ingestion (Later, Real Violent Data)

### Option A: Folder-based single-label ingestion

Use `scripts/build_video_manifest.py`:

```bash
python scripts/build_video_manifest.py \
  --class-dir normal=/path/to/normal \
  --class-dir robbery=/path/to/robbery \
  --class-dir fighting=/path/to/fighting \
  --class-dir shooting=/path/to/shooting \
  --output manifests/multiclass_manifest.csv \
  --infer-duration
```

### Option B: Manifest-first ingestion (recommended for multilabel)

Use CSV with columns:

- `id`
- `video_path`
- `split` (`train|val|test`)
- `label` (single-label mode)
- `labels` (pipe-separated multilabel, e.g. `violent|robbery`)
- `duration_sec` (optional)

Example:

```csv
id,video_path,split,label,labels,duration_sec
clip_001,/data/videos/clip_001.mp4,train,violent,violent|robbery,12.35
```

## 5) Model Backends To Compare

### Stable baseline models (recommended starting point)

- `r3d_18`
- `mc3_18`
- `r2plus1d_18`

### Optional MMAction2 backend

- `mmaction2` (adapter in `video_pipeline/models.py`)
- Template config: `configs/template_mmaction2.yaml`
- You must set `model.mmaction_config_path` to an actual MMAction2 config file.

### Experimental VLM adapters

- `internvideo2_5`
- `videollama3`
- `internvl`
- `hf_backbone`

These are adapter-level integrations and may require official repo-specific fine-tuning recipes for best results.

## 6) Colab Training With Auto-Resume

### Step 1: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone repo and install deps

```bash
cd /content
git clone <your_repo_url> transverse_project_idemia
cd transverse_project_idemia
bash scripts/colab_setup.sh
```

Optional MMAction2 install:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmaction2
```

### Step 3: Launch training with persistent run dir

```bash
python scripts/colab_autoresume.py \
  --config configs/template_ucfcrime_multiclass_from_taxonomy.yaml \
  --run-dir /content/drive/MyDrive/violence_runs/r3d18_multiclass \
  --manifest manifests/bootstrap_multiclass_manifest.csv \
  --class-names-file manifests/ucfcrime_taxonomy.json
```

If Colab disconnects, rerun the same command; it auto-resumes from:

`/content/drive/MyDrive/violence_runs/r3d18_multiclass/checkpoint_last.pt`

## 7) Resume and Checkpoint Policy

Each epoch writes:

- `checkpoint_epoch_XXX.pt`
- `checkpoint_last.pt`
- `checkpoint_best.pt` (best `val macro_f1`)

Manual resume:

```bash
python train_video_classifier.py \
  --config configs/template_binary.yaml \
  --resume runs/binary_r3d18/checkpoint_last.pt
```

## 8) Compare Models

After training multiple runs:

```bash
python scripts/compare_runs.py \
  --run-dir runs/ucfcrime_multiclass_r3d18 \
  --run-dir runs/mmaction2_multiclass \
  --run-dir runs/experimental_vlm \
  --output-csv runs/model_comparison.csv
```

Comparison is ranked by `macro_f1`.

## 9) Team Handover Checklist

1. Keep class naming normalized in manifests (`snake_case` recommended).
2. Re-run taxonomy extraction when new JSON annotation files arrive.
3. Use synthetic manifests only for smoke tests, never for final reporting.
4. Save production checkpoints to persistent storage (Drive/cloud bucket).
5. Track model config + manifest + checkpoint path together for reproducibility.
6. Use `scripts/compare_runs.py` for standardized model comparison reports.
