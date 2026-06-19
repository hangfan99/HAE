import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.era5_128x256_finetune import era5_128x256_finetune
from model.hybrid_vaeformer import HybridVAEformer
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
        if str(model_cfg["model_version"]).startswith("hybrid_"):
            return HybridVAEformer(
                in_chans=model_cfg.get("in_chans", 69),
                out_chans=model_cfg.get("out_chans", 69),
                img_size=tuple(model_cfg.get("img_size", [128, 256])),
                stem_dims=tuple(model_cfg.get("stem_dims", [256, 512, 1024])),
                embed_dim=model_cfg.get("embed_dim", 1024),
                latent_dim=model_cfg.get("latent_dim", 552),
                depth=model_cfg.get("depth", 12),
                num_heads=model_cfg.get("num_heads", 16),
                mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
                qkv_bias=model_cfg.get("qkv_bias", True),
                window=model_cfg.get("window", True),
                window_size=tuple(tuple(x) for x in model_cfg.get("window_size", [[4, 4], [4, 8], [8, 4]])),
                interval=model_cfg.get("interval", 4),
                drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
                sample_posterior=model_cfg.get("sample_posterior", False),
                learnable_pos=model_cfg.get("learnable_pos", True),
            )
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
