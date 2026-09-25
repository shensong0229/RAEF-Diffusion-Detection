# RAEF-Diffusion-Detection
Official implementation for the manuscript submitted to The Visual Computer: Adaptive Reconstruction-Aware Evidence Fusion for Generalisable Diffusion-Generated Image Detection.

## Repository contents

| Path | Purpose |
| --- | --- |
| `core/` | Datasets, transforms, models, losses, metrics, and training utilities |
| `train_supervised.py` | Supervised training entry point |
| `configs/` | Example configurations for single- and dual-branch models |
| `tools/` | Index construction, evaluation, ablation, and visualization scripts |
| `external_baselines/reviewer3_runner/` | Our scripts for recent-baseline evaluation and diagnostic analyses |

This repository contains source code and example configurations only. Datasets, CSV indexes, pretrained VAE files, checkpoints, caches, predictions, and other experiment outputs are not included. Third-party baseline implementations and weights must be obtained from their original projects under their respective licenses.

## Setup

Install a CUDA-compatible PyTorch build and the packages in `requirements.txt`. The reconstruction-aware branch requires a Stable Diffusion 1.5 VAE. Set `FIRE_VAE_DIR` to its local directory if it is not found automatically by `core/models/fire_official.py`. The recent-baseline scripts may also require official third-party implementations or the official CLIP package, which are not bundled here.

Training and evaluation use a CSV index with columns `path`, `label`, and `domain`; `split` is optional. Image paths should resolve relative to the supplied data root or be absolute. Keep training, validation, and unseen-generator test indexes separate. Example configurations contain local checkpoint paths that must be replaced for a new environment.

From the repository root, the main training command has the form:

```powershell
python train_supervised.py --config configs/sfire_crossattn_resnet50.yaml --data_root <image-root> --index_csv <train-validation-index.csv> --spatial_ckpt <spatial-checkpoint> --fire_ckpt <reconstruction-checkpoint> --run_name <run-name>
```

Use `python train_supervised.py --help` and each tool's `--help` for further options. Source indexes, initialization checkpoints, and evaluation bundles establish the exact experimental protocol; a configuration filename alone does not. Match any reported paper result to the corresponding run log and prediction files before release.

The paper-facing dataset and evaluation procedure is described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). In particular, evaluate the reconstruction-aware detector at batch size 1 because its Fourier-processed input uses batch-wide extrema; batch size 1 makes each image's prediction independent of the other images in its batch. The released analysis tools require an explicit checkpoint-protocol label and record the checkpoint SHA-256 digest, rather than inferring the training source from a filename.

Pretrained detector weights are not yet included in this code commit. They must be matched to the final source-domain experiment and published separately before the repository can satisfy a code-**and**-weights release requirement. Do not label a checkpoint as SDV5-trained solely because of its folder name or an evaluation table.
