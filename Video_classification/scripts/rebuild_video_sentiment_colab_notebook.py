#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "Video_Sentiment_Colab_End_to_End.ipynb"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).lstrip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).lstrip("\n").splitlines(keepends=True),
    }


def build_cells() -> list[dict]:
    return [
        md_cell(
            """
            # UCF-Crime Colab Fine-Tuning Notebook

            This notebook assumes you already downloaded the Kaggle dataset into your Colab runtime.
            It now does the following end to end:

            - builds binary, multiclass, and multilabel manifests from `UCFCrime_Train_AnomalyTrain.json`, `UCFCrime_Val_AnomalyTrain.json`, and `UCFCrime_Test_AnomalyTrain.json`
            - matches videos by JSON key to filenames such as `Abuse001_x264.mp4`
            - fine-tunes pretrained torchvision video backbones instead of training from scratch
            - saves checkpoints every epoch and lets you persist runs in Google Drive
            - evaluates saved checkpoints and compares different model backbones
            """
        ),
        code_cell(
            """
            import os
            import sys
            from pathlib import Path

            IN_COLAB = "google.colab" in sys.modules
            print("IN_COLAB =", IN_COLAB)

            if IN_COLAB:
                from google.colab import drive

                drive.mount("/content/drive")
                print("Drive mounted at /content/drive")
            else:
                print("Running outside Colab; Drive mount skipped.")
            """
        ),
        code_cell(
            """
            import os
            import subprocess
            from pathlib import Path

            CANDIDATE_PROJECT_DIRS = [
                Path.cwd(),
                Path.cwd() / "Video_classification",
                Path("/content/transverse_project/Video_classification"),
                Path("/content/transverse_project_idemia/Video_classification"),
                Path("/content/drive/MyDrive/transverse_project/Video_classification"),
            ]

            VC_DIR = None
            for candidate in CANDIDATE_PROJECT_DIRS:
                if (candidate / "train_video_classifier.py").exists():
                    VC_DIR = candidate.resolve()
                    break

            REPO_GIT_URL = ""  # Optional: set if the repo is not already present in Colab.
            if VC_DIR is None:
                if not REPO_GIT_URL:
                    raise FileNotFoundError(
                        "Could not find the Video_classification project directory. "
                        "Clone/upload the repo or set REPO_GIT_URL."
                    )
                clone_root = Path("/content/transverse_project")
                if not clone_root.exists():
                    subprocess.run(["git", "clone", REPO_GIT_URL, str(clone_root)], check=True)
                VC_DIR = (clone_root / "Video_classification").resolve()

            REPO_DIR = VC_DIR.parent
            os.chdir(VC_DIR)
            print("REPO_DIR:", REPO_DIR)
            print("VC_DIR:", VC_DIR)
            print("Working directory:", Path.cwd())
            """
        ),
        code_cell(
            """
            import importlib
            import subprocess
            import sys

            subprocess.run(["bash", "scripts/colab_setup.sh"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"], check=True)

            INSTALL_TRANSFORMERS = False
            if INSTALL_TRANSFORMERS:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "transformers", "accelerate"],
                    check=True,
                )

            required_modules = ["torch", "torchvision", "av", "yaml", "matplotlib"]
            for module_name in required_modules:
                importlib.import_module(module_name)

            import av
            import matplotlib
            import torch
            import torchvision
            import yaml

            print("torch:", torch.__version__)
            print("torchvision:", torchvision.__version__)
            print("pyav:", av.__version__)
            print("pyyaml:", yaml.__version__)
            print("matplotlib:", matplotlib.__version__)
            print("cuda_available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("gpu:", torch.cuda.get_device_name(0))
            else:
                print("GPU not detected. Training will still run, but much slower.")
            """
        ),
        md_cell(
            """
            ## Build The Real UCFCrime Subset

            The Kaggle download cell unzips videos into the Colab runtime. Point `DATASET_ROOT` at the folder
            that contains those videos (using `/content` is usually enough because the builder searches recursively).
            """
        ),
        code_cell(
            """
            import json
            from pathlib import Path

            DATASET_ROOT = Path("/content")  # Change this only if your Kaggle unzip landed elsewhere.

            TRAIN_JSON = VC_DIR / "UCFCrime_Train_AnomalyTrain.json"
            VAL_JSON = VC_DIR / "UCFCrime_Val_AnomalyTrain.json"
            TEST_JSON = VC_DIR / "UCFCrime_Test_AnomalyTrain.json"

            MANIFEST_OUTPUT_DIR = VC_DIR / "manifests" / "ucfcrime_colab_subset"
            MANIFEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            def json_entry_count(path: Path) -> int:
                return len(json.loads(path.read_text(encoding="utf-8-sig")))

            print("DATASET_ROOT:", DATASET_ROOT)
            print("TRAIN_JSON:", TRAIN_JSON, "entries=", json_entry_count(TRAIN_JSON))
            print("VAL_JSON:", VAL_JSON, "entries=", json_entry_count(VAL_JSON))
            print("TEST_JSON:", TEST_JSON, "entries=", json_entry_count(TEST_JSON))
            print("MANIFEST_OUTPUT_DIR:", MANIFEST_OUTPUT_DIR)
            """
        ),
        code_cell(
            """
            import json
            import subprocess
            import sys

            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_ucfcrime_subset_manifests.py",
                    "--dataset-root",
                    str(DATASET_ROOT),
                    "--train-json",
                    str(TRAIN_JSON),
                    "--val-json",
                    str(VAL_JSON),
                    "--test-json",
                    str(TEST_JSON),
                    "--output-dir",
                    str(MANIFEST_OUTPUT_DIR),
                    "--strict",
                ],
                check=True,
            )

            BINARY_MANIFEST = MANIFEST_OUTPUT_DIR / "ucfcrime_binary_manifest.csv"
            MULTICLASS_MANIFEST = MANIFEST_OUTPUT_DIR / "ucfcrime_multiclass_manifest.csv"
            MULTILABEL_MANIFEST = MANIFEST_OUTPUT_DIR / "ucfcrime_multilabel_manifest.csv"
            TAXONOMY_JSON = MANIFEST_OUTPUT_DIR / "ucfcrime_taxonomy.json"
            BUILD_SUMMARY_JSON = MANIFEST_OUTPUT_DIR / "build_summary.json"

            build_summary = json.loads(BUILD_SUMMARY_JSON.read_text(encoding="utf-8"))
            print(json.dumps(build_summary, indent=2))
            """
        ),
        code_cell(
            """
            import json
            import matplotlib.pyplot as plt

            build_summary = json.loads(BUILD_SUMMARY_JSON.read_text(encoding="utf-8"))

            def plot_counts(counts_by_split: dict, title: str) -> None:
                for split_name, label_counts in counts_by_split.items():
                    labels = list(label_counts.keys())
                    values = [label_counts[label] for label in labels]
                    plt.figure(figsize=(12, 4))
                    plt.bar(labels, values)
                    plt.title(f"{title} - {split_name}")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.show()

            plot_counts(build_summary["binary_counts_by_split"], "Binary label distribution")
            plot_counts(build_summary["multiclass_counts_by_split"], "Multiclass label distribution")

            missing_total = sum(len(values) for values in build_summary["missing_keys"].values())
            print("Missing matches:", missing_total)
            if missing_total:
                print(json.dumps(build_summary["missing_keys"], indent=2))

            duplicate_total = len(build_summary["duplicate_matches"])
            print("Duplicate matches:", duplicate_total)
            if duplicate_total:
                print(json.dumps(build_summary["duplicate_matches"], indent=2))
            """
        ),
        md_cell(
            """
            ## Training

            The templates already use `pretrained: true`, so the notebook now fine-tunes pretrained backbones.
            You can train one model or several back-to-back and keep every checkpoint in Drive.
            """
        ),
        code_cell(
            """
            from pathlib import Path

            TASK = "multiclass"  # "binary" | "multiclass" | "multilabel"
            MODEL_NAMES = ["r3d_18"]  # also try: ["r3d_18", "mc3_18", "r2plus1d_18"]

            RUN_TRAINING = False  # Set True when you are ready to train.
            VERIFY_PRETRAINED_DOWNLOAD = False

            EPOCHS = 10
            BATCH_SIZE = 2
            NUM_WORKERS = 2
            LEARNING_RATE = 1e-4
            NUM_FRAMES = 16
            IMAGE_SIZE = 112
            CLIP_DURATION_SEC = 2.0
            FREEZE_BACKBONE = False
            DEVICE = None  # e.g. "cuda:0" or "cpu"

            if IN_COLAB and Path("/content/drive/MyDrive").exists():
                RUNS_BASE_DIR = Path("/content/drive/MyDrive/ucfcrime_runs")
            else:
                RUNS_BASE_DIR = VC_DIR / "runs" / "ucfcrime_runs"
            RUNS_BASE_DIR.mkdir(parents=True, exist_ok=True)

            TASK_TO_TEMPLATE = {
                "binary": VC_DIR / "config" / "template_binary.yaml",
                "multiclass": VC_DIR / "config" / "template_ucfcrime_multiclass_from_taxonomy.yaml",
                "multilabel": VC_DIR / "config" / "template_ucfcrime_multilabel_from_taxonomy.yaml",
            }
            TASK_TO_MANIFEST = {
                "binary": BINARY_MANIFEST,
                "multiclass": MULTICLASS_MANIFEST,
                "multilabel": MULTILABEL_MANIFEST,
            }

            print("TASK:", TASK)
            print("MODEL_NAMES:", MODEL_NAMES)
            print("RUNS_BASE_DIR:", RUNS_BASE_DIR)
            print("Manifest for task:", TASK_TO_MANIFEST[TASK])
            print("Base config:", TASK_TO_TEMPLATE[TASK])
            """
        ),
        code_cell(
            """
            import torch

            from video_pipeline.models import ModelConfig, build_model

            if VERIFY_PRETRAINED_DOWNLOAD:
                for model_name in MODEL_NAMES:
                    print(f"Downloading/loading pretrained weights for {model_name} ...")
                    model = build_model(
                        ModelConfig(
                            name=model_name,
                            pretrained=True,
                            dropout=0.2,
                            freeze_backbone=FREEZE_BACKBONE,
                        ),
                        num_outputs=2,
                    )
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print("Pretrained weights are available in the runtime cache.")
            else:
                print("Skipping explicit pretrained-weight warmup. Training will download weights on first use.")
            """
        ),
        code_cell(
            """
            import json
            import subprocess
            import sys

            import yaml

            def load_yaml(path: Path) -> dict:
                with path.open("r", encoding="utf-8") as handle:
                    return yaml.safe_load(handle)

            def dump_yaml(payload: dict, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(payload, handle, sort_keys=False)

            def choose_resume_checkpoint(run_dir: Path) -> Path | None:
                last_path = run_dir / "checkpoint_last.pt"
                if last_path.exists():
                    return last_path
                epoch_checkpoints = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
                if epoch_checkpoints:
                    return epoch_checkpoints[-1]
                return None

            def prepare_runtime_config(task: str, model_name: str) -> tuple[Path, Path, Path | None]:
                run_dir = RUNS_BASE_DIR / f"{task}_{model_name}"
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg = load_yaml(TASK_TO_TEMPLATE[task])
                cfg["manifest_path"] = str(TASK_TO_MANIFEST[task])
                cfg.setdefault("model", {})
                cfg["model"]["name"] = model_name
                cfg["model"]["pretrained"] = True
                cfg["model"]["freeze_backbone"] = FREEZE_BACKBONE

                cfg.setdefault("training", {})
                cfg["training"]["epochs"] = EPOCHS
                cfg["training"]["batch_size"] = BATCH_SIZE
                cfg["training"]["num_workers"] = NUM_WORKERS

                cfg.setdefault("optimizer", {})
                cfg["optimizer"]["lr"] = LEARNING_RATE

                cfg.setdefault("data", {})
                cfg["data"]["num_frames"] = NUM_FRAMES
                cfg["data"]["image_size"] = IMAGE_SIZE
                cfg["data"]["clip_duration_sec"] = CLIP_DURATION_SEC

                cfg.setdefault("checkpoint", {})
                cfg["checkpoint"]["output_dir"] = str(run_dir)
                cfg["checkpoint"]["resume_from"] = None

                if cfg.get("scheduler", {}).get("name") == "cosine":
                    cfg["scheduler"]["t_max"] = EPOCHS

                if task == "binary":
                    cfg["class_names"] = ["normal", "violent"]
                    cfg.pop("class_names_file", None)
                    cfg.pop("class_names_key", None)
                elif task == "multiclass":
                    cfg["class_names_file"] = str(TAXONOMY_JSON)
                    cfg["class_names_key"] = "multiclass_classes"
                elif task == "multilabel":
                    cfg["class_names_file"] = str(TAXONOMY_JSON)
                    cfg["class_names_key"] = "multilabel_classes"
                else:
                    raise ValueError(f"Unsupported task: {task}")

                runtime_config = run_dir / "runtime_config.yaml"
                dump_yaml(cfg, runtime_config)
                resume_checkpoint = choose_resume_checkpoint(run_dir)
                return run_dir, runtime_config, resume_checkpoint

            def build_train_command(task: str, model_name: str) -> tuple[Path, list[str]]:
                run_dir, runtime_config, resume_checkpoint = prepare_runtime_config(task, model_name)
                command = [
                    sys.executable,
                    "train_video_classifier.py",
                    "--config",
                    str(runtime_config),
                ]
                if resume_checkpoint is not None:
                    command.extend(["--resume", str(resume_checkpoint)])
                if DEVICE:
                    command.extend(["--device", DEVICE])
                return run_dir, command
            """
        ),
        code_cell(
            """
            LAUNCHED_RUN_DIRS = []

            for model_name in MODEL_NAMES:
                run_dir, command = build_train_command(TASK, model_name)
                LAUNCHED_RUN_DIRS.append(run_dir)
                print("\\nRun directory:", run_dir)
                print("Command:", " ".join(command))
                if RUN_TRAINING:
                    subprocess.run(command, check=True)

            if RUN_TRAINING:
                print("\\nTraining complete. Checkpoints are stored in:")
                for run_dir in LAUNCHED_RUN_DIRS:
                    print(" -", run_dir)
            else:
                print("\\nRUN_TRAINING=False, so the notebook only prepared configs and printed commands.")
            """
        ),
        code_cell(
            """
            import shutil

            if "LAUNCHED_RUN_DIRS" in globals() and LAUNCHED_RUN_DIRS:
                EXPORT_RUN_DIR = LAUNCHED_RUN_DIRS[-1]
            else:
                EXPORT_RUN_DIR = RUNS_BASE_DIR / f"{TASK}_{MODEL_NAMES[0]}"

            DOWNLOAD_TO_LOCAL = False

            if not EXPORT_RUN_DIR.exists():
                print("Run directory not found:", EXPORT_RUN_DIR)
            else:
                checkpoint_files = sorted(EXPORT_RUN_DIR.glob("checkpoint*.pt"))
                print("Checkpoint files:")
                for checkpoint_path in checkpoint_files:
                    print(" -", checkpoint_path.name)

                archive_path = Path(
                    shutil.make_archive(str(EXPORT_RUN_DIR), "zip", root_dir=str(EXPORT_RUN_DIR))
                )
                print("Archive written to:", archive_path)

                if IN_COLAB and DOWNLOAD_TO_LOCAL:
                    from google.colab import files

                    files.download(str(archive_path))
            """
        ),
        code_cell(
            """
            import json
            import subprocess
            import sys

            import matplotlib.pyplot as plt

            EVAL_TASK = TASK
            EVAL_MODEL_NAME = MODEL_NAMES[0]
            EVAL_RUN_DIR = RUNS_BASE_DIR / f"{EVAL_TASK}_{EVAL_MODEL_NAME}"
            EVAL_CONFIG = EVAL_RUN_DIR / "runtime_config.yaml"
            BEST_CKPT = EVAL_RUN_DIR / "checkpoint_best.pt"

            if not EVAL_CONFIG.exists():
                print("Runtime config not found:", EVAL_CONFIG)
                print("Run the training-preparation cell first.")
            elif not BEST_CKPT.exists():
                print("Best checkpoint not found:", BEST_CKPT)
                print("Train the model first or point EVAL_RUN_DIR at an existing run.")
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "test_video_classifier.py",
                        "--config",
                        str(EVAL_CONFIG),
                        "--checkpoint",
                        str(BEST_CKPT),
                        "--split",
                        "test",
                    ],
                    check=True,
                )

                metrics_path = EVAL_RUN_DIR / "metrics_test_from_checkpoint_best.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                print(json.dumps(metrics, indent=2))

                per_class = metrics.get("per_class", [])
                if per_class:
                    classes = [row["class_name"] for row in per_class]
                    f1_scores = [row["f1"] for row in per_class]
                    plt.figure(figsize=(12, 4))
                    plt.bar(classes, f1_scores)
                    plt.title(f"Per-class F1 ({EVAL_TASK})")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.show()
            """
        ),
        code_cell(
            """
            import subprocess
            import sys

            if "LAUNCHED_RUN_DIRS" in globals() and len(LAUNCHED_RUN_DIRS) >= 2:
                RUN_DIRS_TO_COMPARE = LAUNCHED_RUN_DIRS
            else:
                RUN_DIRS_TO_COMPARE = []

            if len(RUN_DIRS_TO_COMPARE) < 2:
                print("Add at least two trained run directories to RUN_DIRS_TO_COMPARE to compare models.")
            else:
                comparison_csv = RUNS_BASE_DIR / f"{TASK}_model_comparison.csv"
                command = [sys.executable, "scripts/compare_runs.py"]
                for run_dir in RUN_DIRS_TO_COMPARE:
                    command.extend(["--run-dir", str(run_dir)])
                command.extend(["--output-csv", str(comparison_csv)])
                subprocess.run(command, check=True)
                print("Comparison CSV:", comparison_csv)
            """
        ),
        md_cell(
            """
            ## Notes

            - `r3d_18`, `mc3_18`, and `r2plus1d_18` are all supported pretrained torchvision backbones for this notebook.
            - The first pretrained run downloads model weights automatically; later runs reuse the local cache.
            - `train_video_classifier.py` saves `checkpoint_epoch_XXX.pt`, `checkpoint_last.pt`, `checkpoint_best.pt`, and `history.json` into each run directory.
            - Keeping `RUNS_BASE_DIR` under `/content/drive/MyDrive/...` makes the checkpoints survive Colab disconnects.
            - The export cell also creates a `.zip` archive that you can download to your local machine if you want an offline copy.
            """
        ),
    ]


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook["cells"] = build_cells()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Rebuilt {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
