# Reproducing the evaluation protocol

This page describes the inputs and code paths required to repeat the paper's evaluation. It does not substitute for the original training log and checkpoint metadata: those must accompany each released detector weight.

## Data and splits

The SDV5-source protocol uses 100,000 real and 100,000 SDV5-generated training images, plus 10,000 real and 10,000 SDV5-generated validation images. The unseen-generator test comprises ten domains with 1,000 real and 1,000 generated images each, with no real-image overlap across domains. No target-domain sample is used for training, checkpoint selection, threshold selection, or calibration.

The index schema is `path,label,domain,split`, where `label=0` is real and `label=1` is fake. The `split` column is optional for a single-split index but is needed for combined train/validation indexes. `tools/build_paper_sdv5_source_index.py` constructs and verifies a fixed source index; `tools/build_dragon_eval_csvs_unique_real.py` prepares unique-real evaluation indexes. Supply the image roots yourself. Image data and indexes are not redistributed here.

## Training

The main entry point is `train_supervised.py`. The `configs/paper_sdv5_spatial_resnet50.yaml` and `configs/paper_sdv5_fire_resnet50.yaml` files describe the two branch models. The dual-branch model requires both branch checkpoint paths, passed with `--spatial_ckpt` and `--fire_ckpt` or specified in a matching configuration. The capacity-controlled comparison is implemented by `tools/run_paper_sdv5_ablation_pipeline.py` with `configs/paper_sdv5_fusion_capacity_ablation.yaml`; it is a separately trained, same-initialization comparison, not the fixed-checkpoint test-time intervention.

Keep the following with each published checkpoint: source-index hash and class counts, exact configuration and command line, initialization checkpoint hashes, random seed, selected epoch, model name, code commit, and the validation metric used for selection. The example configurations are not evidence that an unavailable cloud run used those exact settings.

## Ten-domain inference

Use the same ten test indexes for all methods. For the full detector, `tools/eval_zs_suite.py` evaluates one checkpoint across a directory of per-domain CSV files:

```powershell
python tools/eval_zs_suite.py --ckpt <full-model-checkpoint> --data_root <test-image-root> --csv_dir <ten-domain-index-dir> --model_name sfire_crossattn_resnet50 --image_size 256 --batch_size 1 --num_workers 1 --out_csv <per-domain-metrics.csv>
```

For aligned per-image full-model, equal-weight-routing, and no-anomaly-guidance predictions, run `external_baselines/reviewer3_runner/reviewer3_test_time_ablation.py` with explicit `--checkpoint`, `--csv-dir`, `--data-root`, `--output-dir`, and `--checkpoint-protocol`. These are **inference-time switches on one fixed checkpoint**, not separately retrained ablation models. `reviewer3_routing_eval.py` exports aligned branch predictions and weights. `analyze_reviewer3_routing.py` checks that both inputs refer to the same checkpoint before computing routing and uncertainty statistics. Use `--checkpoint-protocol SDV5-source` only for a checkpoint whose training record actually establishes SDV5-source training.

## Metrics and uncertainty

`tools/summarize_predictions.py` reads the ten per-image prediction CSVs, checks for 2,000 images and balanced classes in each domain, and reports domain-level and macro AUC, AP, ACC, F1, and EER. ACC and F1 use a fake-class probability threshold fixed at 0.50. AUC, AP, and EER use the continuous score. Its descriptive 95% Student-t intervals are across **ten domain-level metrics**, not 20,000 independent image-level observations:

```powershell
python tools/summarize_predictions.py --prediction-dir <per-image-prediction-dir> --score-column prob_full --output-json <metric-summary.json>
```

The routing analysis script additionally supports paired, within-domain/class bootstrap intervals when aligned per-image results are available. Do not substitute those image-level intervals for the across-domain interval reported in the paper without changing the description.

## External methods and weights

SPAI, GFRE, and single-class attribution runners are under `external_baselines/reviewer3_runner/`. The official SPAI implementation, official CLIP implementation and weights, and all other third-party detector weights are separate dependencies; obtain them from their authors and observe their licenses. The repository contains neither the Stable Diffusion VAE nor third-party benchmark images.
