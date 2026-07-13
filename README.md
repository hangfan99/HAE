# HAE AE-KL Training Pipeline

This repo now contains a runnable AE-KL training pipeline migrated from `AE/train_AE_KL.py`, with a cleaner structure inspired by `forecast`:

- `configs/ae_kl.yaml`: experiment config
- `train.py`: training entrypoint
- `utils/builder.py`: dataset/model/optimizer builder
- `trainers/ae_kl_trainer.py`: train/validate/checkpoint loop
- `dataset/era5_128x256_finetune.py`: ERA5 dataset loader
- `model/vaeformer.py` + `model/vit_nlc.py`: VAEformer model

## Run

```bash
bash scripts/run_train_ae_kl.sh
```

or

```bash
python train.py --cfg configs/ae_kl.yaml --outdir output --world_size 1 --per_cpus 4
```

## Resume

```bash
python train.py --cfg configs/ae_kl.yaml --outdir output --resume --resume_dir <existing_run_dir>
```

If `flash_attn` is unavailable, attention automatically falls back to standard PyTorch attention.

## Current HAE Handoff

This repo is currently used for ERA5 hierarchical autoencoder experiments. The active model is `HybridHier2VAEformer` in `model/hybrid_vaeformer.py`, constructed from `utils/builder.py`. The task is moving to another cluster, so this section records the current state and the recommended next run.

### Recommended Baseline

Use this config as the main baseline:

```text
configs/ae_kl_hybrid_hier2_34_balanced_ae20.yaml
```

It uses the same latent layout as the best-performing current HAE:

```text
z_bottom: 34 x 32 x 64
z_top:    34 x 8 x 16
total latent elements: 73,984
parameters: about 232M
```

Key architecture choices:

```yaml
encoder_depths: [1, 2, 2, 4]
decoder_depths: [1, 1, 8, 6]
bottom_level: 2              # implicit default: bottom comes from 32 x 64 feature level
bottom_fusion: residual_multiscale
bottom_fuse_block: residual_3x3
bottom_drop_period: 0
loss:
  enable_kl: false
```

The decoder fuses bottom features at `32 x 64`, then also sends bottom information to a finer decoder stage through the `residual_multiscale` path. This is the configuration that reduced reconstruction error after strengthening the bottom decoder.

Known local result from the earlier balanced run:

```text
experiment: AE_hybrid_hier2_34_balanced_ae20
best validation mix error: 0.067550
log/checkpoint dir: output/AE_hybrid_hier2_34_balanced_ae20/
```

If this checkpoint is not present after migration, retrain from the config above.

### Bottom-Drop Ablation

Use this config only for the ablation where the model is forced to make the top latent useful:

```text
configs/ae_kl_hybrid_hier2_34_balanced_bottomdrop_ae20.yaml
```

It keeps the same architecture as the balanced baseline, but enables deterministic bottom drop:

```yaml
bottom_drop_period: 5   # every 5 train steps, z_bottom is zeroed; about 20% top-only steps
batch_size: 8
```

This is intended to answer whether top can carry useful coarse information. It is not the current best pure-reconstruction setting. Expect reconstruction to be worse than no-drop early in training.

Local partial run state before migration:

```text
experiment: AE_hybrid_hier2_34_balanced_bottomdrop_ae20
checkpoint dir: output/AE_hybrid_hier2_34_balanced_bottomdrop_ae20/
local log reached epoch 3, step 8600/10135
```

Batch 16 OOMed on the shared old node; batch 8 was used there. On a clean cluster with enough memory, batch 16 should be tested again.

### Avoided Direction

The level-3 bottom latent config is kept for reference, but should not be the next main run:

```text
configs/ae_kl_hybrid_hier2_136b16_34t8_fast_ae20.yaml
```

It changes the bottom latent from `34 x 32 x 64` to `136 x 16 x 32`. Although the element count is similar, it loses spatial detail too early and produced noticeably worse reconstruction. It is also not the cleanest answer to the top-latent issue.

Current recommendation:

- Keep bottom at `34 x 32 x 64`.
- Keep the balanced decoder `[1, 1, 8, 6]` as the stable baseline.
- Use bottom-drop ablation to test whether top carries information.
- If increasing model size, do it modestly around the bottom decoder/fusion path, not by moving bottom to `16 x 32`.

### Running On A New Cluster

The parent directory has the environment file used on the old cluster:

```text
../environment.yml
```

The dataset paths in configs currently point to:

```yaml
data_dir: huawei_100p:s3://ai4earth/era5_np128x256
```

On the new cluster, either make this Petrel/S3 path work or update `dataset.train.data_dir` and `dataset.valid.data_dir` in the selected config.

Single-node 4-GPU Slurm-style command:

```bash
PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))
mkdir -p out
srun -p <partition> --quotatype=<quotatype> --job-name=hae_balanced \
  --ntasks-per-node=4 --cpus-per-task=1 -N 1 \
  -o ./out/train_%j.out --gres=gpu:4 --kill-on-bad-exit=1 \
  python -u train.py \
  --cfg configs/ae_kl_hybrid_hier2_34_balanced_ae20.yaml \
  --outdir output \
  --init_method tcp://127.0.0.1:$PORT \
  --per_cpus 4 \
  --world_size 4
```

For the bottom-drop ablation, switch only the config:

```bash
--cfg configs/ae_kl_hybrid_hier2_34_balanced_bottomdrop_ae20.yaml
```

`slurm_train.sh` currently has a hard-coded `yaml=` value. If using that script on the new cluster, edit `yaml`, `partition`, `quotatype`, `job_name`, and `gpus` before submitting.

### Checkpoints And Logs

Training writes fixed experiment directories under `output/`:

```text
output/<experiment.name>/
  best.pth
  config_resolved.yaml
  train.log
  tb/
```

Configs are set to save only the best checkpoint:

```yaml
save_best_checkpoint: true
save_final_checkpoint: false
save_latest_snapshot: false
save_optimizer_in_snapshot: false
```

Useful checks:

```bash
tail -120 output/<experiment.name>/train.log
tail -120 out/train_<jobid>.out
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -P
find . -maxdepth 3 -type f -size +100M
```

### Evaluation Notebook

Manual reconstruction evaluation is in:

```text
HAE_hier2_recon_manual_eval.ipynb
```

It supports selecting cases manually and comparing:

- HAE top-only reconstruction
- HAE top+bottom reconstruction
- absolute `top+bottom - top`
- signed reconstruction error against truth
- DC-AE comparison from `AE_KL_hybrid_1024_16_full/best`
- MAE and bias summaries

Use this notebook after each candidate checkpoint is available.

### Implementation Notes

- `bottom_drop_period` is deterministic from `global_step`. Do not replace it with random per-rank branching, or DDP can diverge.
- Under Slurm, `utils/misc.py` now uses a local `file://.dist_init/torch_<job>_<step>` init path for DDP. `.dist_init/` is runtime state and should not be committed.
- The final output stays linear. ERA5 variables are standardized and can be negative; do not add final `ReLU`, `SiLU`, or `BatchNorm`.
- Dataset workers can leave child processes if interrupted. The trainer/dataset cleanup path is important; keep `refresh_dataloader_each_epoch: true`.
- Runtime outputs such as `.dist_init/`, `out/`, `output_*`, and `eval_outputs/` should stay out of commits unless explicitly archiving results.

### Useful Files

```text
model/hybrid_vaeformer.py                              # HAE model and fusion logic
utils/builder.py                                       # model construction from YAML
trainers/ae_kl_trainer.py                              # train/validation/checkpoint loop
model/AE_2D_v2.py                                      # reconstruction/KL loss
configs/ae_kl_hybrid_hier2_34_balanced_ae20.yaml       # recommended baseline
configs/ae_kl_hybrid_hier2_34_balanced_bottomdrop_ae20.yaml  # top-latent ablation
configs/ae_kl_hybrid_hier2_136b16_34t8_fast_ae20.yaml  # failed/avoid as main route
HAE_hier2_recon_manual_eval.ipynb                      # manual reconstruction eval
slurm_train.sh                                         # old-cluster launch helper
```
