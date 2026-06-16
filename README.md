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
