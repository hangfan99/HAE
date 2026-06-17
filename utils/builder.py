import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.era5_128x256_finetune import era5_128x256_finetune
from model.vaeformer import VAEformer


class ConfigBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def _dataset_cfg(self, split):
        return dict(self.cfg["dataset"].get(split, {}))

    def build_dataset(self, split):
        params = self._dataset_cfg(split)
        return era5_128x256_finetune(split=split, **params)

    def build_dataloader(self, split, num_workers, distributed):
        trainer_cfg = self.cfg["trainer"]
        batch_size_key = f"{split}_batch_size"
        batch_size = trainer_cfg.get(batch_size_key, trainer_cfg["batch_size"])
        dataset = self.build_dataset(split)

        if distributed:
            sampler = DistributedSampler(dataset, shuffle=(split == "train"))
            shuffle = False
        else:
            sampler = None
            shuffle = split == "train"

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        return loader, dataset, sampler

    def build_model(self):
        model_cfg = self.cfg["model"]
        return VAEformer(
            model_version=model_cfg["model_version"],
            sample_posterior=model_cfg.get("sample_posterior", True),
            patch_embed_type=model_cfg.get("patch_embed_type", "baseline"),
            patch_embed_hidden_dims=model_cfg.get("patch_embed_hidden_dims", None),
        )

    def build_optimizer(self, model):
        optim_cfg = self.cfg["optimizer"]
        return torch.optim.AdamW(
            model.parameters(),
            lr=optim_cfg["lr"],
            weight_decay=optim_cfg["weight_decay"],
        )
