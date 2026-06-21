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

## Current Codex Handoff

This repo is currently being used to train ERA5 autoencoders derived from `/mnt/petrelfs/fanhang/AE` VAEformer code. The active direction is a two-level hierarchical AE inspired by VQ-VAE2-style top/bottom latents, while keeping the existing dataset, trainer, validation metrics, and AE-KL training pipeline.

### Active Model

The default `slurm_train.sh` target is now:

```bash
configs/ae_kl_hybrid_hier2_34_full.yaml
```

This builds `HybridHier2VAEformer` in `model/hybrid_vaeformer.py` through `utils/builder.py`. The intended latent shapes are:

```text
z_bottom: 34 x 32 x 64   # higher-resolution detail latent
z_top:    34 x 8 x 16    # low-resolution coarse/global latent
```

Architecture summary:

```text
input 69 x 128 x 256

encoder:
  DownBlock 69 -> 256,     64 x 128, + StageViT depth 1
  DownBlock 256 -> 512,    32 x 64,  + StageViT depth 2
  q_bottom: Conv 512 -> 2 * 34

  DownBlock 512 -> 768,    16 x 32,  + StageViT depth 2
  DownBlock 768 -> 1024,   8 x 16,   + StageViT depth 4
  q_top: Conv 1024 -> 2 * 34

decoder:
  z_top -> Conv 34 -> 1024, + StageViT depth 4
  UpBlock 1024 -> 768,      16 x 32, + StageViT depth 2
  UpBlock 768 -> 512,       32 x 64
  add-fuse bottom: d2 + bottom_scale * bottom_proj(z_bottom)
  StageViT depth 2
  UpBlock 512 -> 256,       64 x 128, + StageViT depth 1
  UpBlock 256 -> 69,        128 x 256, final linear output
```

Important details:

- Bottom fusion is additive, not concat: `d2 = d2 + bottom_scale * bottom_proj(z_bottom)`.
- `bottom_scale` is learnable and initialized to `0.1`.
- Every `bottom_drop_period` steps, the trainer passes `global_step` and the model sets `z_bottom` to all zeros. Default is `bottom_drop_period: 4`, so 1 in 4 steps trains top-only coarse reconstruction.
- The final output block must stay linear. Do not add `SiLU`, `ReLU`, or final `BatchNorm` to the 69-channel output, because ERA5 data is standardized and contains negative values.

### Loss

`model/AE_2D_v2.py::Mix_loss` supports both the old single posterior and the new dict posterior:

```python
posteriors = {"bottom": q_bottom, "top": q_top}
loss = recon_loss + kl_weights["bottom"] * KL(q_bottom) + kl_weights["top"] * KL(q_top)
```

Current full config uses:

```yaml
loss:
  kl_weight: 1.0e-6
  kl_weights:
    bottom: 1.0e-7
    top: 1.0e-6
  enable_kl: true
```

The bottom KL is intentionally smaller because `34*32*64` has many more latent elements than `34*8*16`.

### Training

Use Slurm. Do not run training directly on a compute node.

Default full training:

```bash
bash slurm_train.sh
```

Equivalent explicit command:

```bash
CFG=configs/ae_kl_hybrid_hier2_34_full.yaml bash slurm_train.sh
```

The script writes submit output to:

```text
out/submit_hae_ae_hybrid_hier2_34.out
```

and Slurm training logs to:

```text
out/train_<jobid>.out
```

Checkpoints are written to a fixed directory, without timestamp subdirectories:

```text
output/AE_KL_hybrid_hier2_34_full/
```

The full config saves only `best.pth` by default:

```yaml
save_best_checkpoint: true
save_final_checkpoint: false
save_latest_snapshot: false
```

### Smoke Test

Use the smoke config before full runs after code changes:

```bash
QUOTATYPE=spot \
CFG=configs/ae_kl_hybrid_hier2_34_smoke.yaml \
OUTDIR=output_spot_test \
JOB_NAME=hae_ae_hier2_add_smoke \
GPUS=4 CPUS=2 \
bash slurm_train.sh
```

Expected behavior:

- 4 distributed train steps.
- Validation runs once.
- `sacct` state should be `COMPLETED` with `ExitCode 0:0`.
- No checkpoint should be saved by the smoke config.

Useful checks:

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -P
tail -120 out/train_<jobid>.out
find . -maxdepth 3 -type f -size +100M
```

Last known good smoke validation for add-fusion/bottom-zero schedule:

```text
job 9811041: COMPLETED 0:0
trained 4 steps and ran validation successfully
```

### Known Pitfalls

- Earlier DDP failure: random per-rank latent dropout caused `bottom_moments.weight has been marked as ready twice`. The fix is deterministic `global_step`-based bottom zeroing and `find_unused_parameters=False`.
- Do not re-enable random per-rank top-only/full branching inside model forward. If top-only training is changed, keep the branch synchronized across ranks.
- Dataset objects spawn many child processes. The trainer and dataset include explicit close/cleanup logic. Keep `refresh_dataloader_each_epoch: true` for full training and close validation loaders after every validation.
- Phoenix `srun` may create small `batchscript-*` files in the repo root. These are runtime artifacts and can be deleted.
- Storage is limited. Avoid saving optimizer snapshots or multiple checkpoints unless explicitly needed. Watch for large files with `find . -maxdepth 3 -type f -size +100M`.

### Useful Files

```text
model/hybrid_vaeformer.py                  # HybridVAEformer and HybridHier2VAEformer
configs/ae_kl_hybrid_hier2_34_full.yaml    # active full config
configs/ae_kl_hybrid_hier2_34_smoke.yaml   # fast Slurm smoke config
model/AE_2D_v2.py                          # Mix_loss with dict posterior support
trainers/ae_kl_trainer.py                  # train/validation/checkpoint loop
utils/builder.py                           # model construction
slurm_train.sh                             # default Slurm entrypoint
```
