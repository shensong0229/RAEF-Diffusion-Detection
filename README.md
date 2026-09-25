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

From the repository root, a training command has the form:

```powershell
python train_supervised.py --config configs/sfire_crossattn_resnet50.yaml --data_root <image-root> --index_csv <train-validation-index.csv> --spatial_ckpt <spatial-checkpoint> --fire_ckpt <reconstruction-checkpoint> --run_name <run-name>
```

This command demonstrates the interface; it is not a claim that the named configuration reproduces Table 2. Some `sfire_crossattn_resnet50*.yaml` files document earlier ADM-initialized runs, while the `paper_sdv5_*.yaml` files are SDV5-oriented examples. The exact protocol of a reported result requires its source-index hash, initialization-checkpoint hashes, full configuration, training log, and predictions. Match these records before treating an example as the final cloud experiment. Use `python train_supervised.py --help` and each tool's `--help` for further options.

The paper-facing dataset and evaluation procedure is described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). In particular, evaluate the reconstruction-aware detector at batch size 1 because its Fourier-processed input uses batch-wide extrema; batch size 1 makes each image's prediction independent of the other images in its batch. The released analysis tools require an explicit checkpoint-protocol label and record the checkpoint SHA-256 digest, rather than inferring the training source from a filename.

Pretrained detector weights are not yet included in this code commit. They must be matched to the final source-domain experiment and published separately before the repository can satisfy a code-**and**-weights release requirement. Do not label a checkpoint as SDV5-trained solely because of its folder name or an evaluation table.

## Attribution

`core/models/fire_official.py` integrates and extends the [FIRE implementation](https://github.com/Chuchad/FIRE), which is MIT-licensed; its original author copyright notice is preserved in `LICENSE`. The SPAI runner is an adapter for the [official SPAI project](https://github.com/mever-team/spai), not a redistribution of SPAI model code or weights. Obtain other external implementations and pretrained components from their respective authors under their own licenses.
