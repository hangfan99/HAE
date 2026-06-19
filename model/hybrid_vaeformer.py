import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .vit_nlc import Block, QuickGELU, get_2d_sincos_pos_embed


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        return self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.]).to(device=self.parameters.device)
        if other is None:
            return 0.5 * torch.mean(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.mean(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def mode(self):
        return self.mean


class ConvBNAct(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1, act=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_chans, out_chans, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_chans),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpaceToChannelShortcut(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Conv2d(in_chans * 4, out_chans, 1, bias=False)

    def forward(self, x):
        return self.proj(self.unshuffle(x))


class ChannelToSpaceShortcut(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans * 4, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.proj(x))


class DownBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct(in_chans, out_chans, kernel_size=3, stride=2),
            ConvBNAct(out_chans, out_chans, kernel_size=3, stride=1, act=False),
        )
        self.shortcut = SpaceToChannelShortcut(in_chans, out_chans)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_chans, out_chans * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_chans),
            nn.SiLU(inplace=True),
            ConvBNAct(out_chans, out_chans, kernel_size=3, stride=1, act=False),
        )
        self.shortcut = ChannelToSpaceShortcut(in_chans, out_chans)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))


class HybridVAEformer(nn.Module):
    def __init__(
        self,
        in_chans=69,
        out_chans=69,
        img_size=(128, 256),
        stem_dims=(256, 512, 1024),
        embed_dim=1024,
        latent_dim=552,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        window=True,
        window_size=((4, 4), (4, 8), (8, 4)),
        interval=4,
        drop_path_rate=0.0,
        sample_posterior=False,
        learnable_pos=True,
    ):
        super().__init__()
        if len(stem_dims) == 0:
            raise ValueError("stem_dims must contain at least one channel size.")
        if stem_dims[-1] != embed_dim:
            raise ValueError(f"Last stem dim {stem_dims[-1]} must equal embed_dim {embed_dim}.")

        downsample = 2 ** len(stem_dims)
        if img_size[0] % downsample != 0 or img_size[1] % downsample != 0:
            raise ValueError(f"img_size {img_size} must be divisible by {downsample}.")

        self.sample_posterior = sample_posterior
        self.img_size = tuple(img_size)
        self.patch_shape = (img_size[0] // downsample, img_size[1] // downsample)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        dims = [in_chans] + list(stem_dims)
        self.stem = nn.Sequential(*[DownBlock(dims[i], dims[i + 1]) for i in range(len(stem_dims))])

        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_shape, cls_token=False)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            requires_grad=learnable_pos,
        )

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            which_win = min(i % interval, len(window_size) - 1)
            block_window = window_size[which_win] if ((i + 1) % interval != 0) else self.patch_shape
            self.blocks.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    window_size=block_window,
                    window=((i + 1) % interval != 0) if window else False,
                    rel_pos_spatial=False,
                    act_layer=QuickGELU,
                )
            )
        self.norm = norm_layer(embed_dim)

        self.to_moments = nn.Conv2d(embed_dim, 2 * latent_dim, 1)
        self.from_latent = nn.Conv2d(latent_dim, embed_dim, 1)

        decode_dims = list(reversed(stem_dims))
        up_layers = []
        for in_dim, out_dim in zip(decode_dims, decode_dims[1:] + [out_chans]):
            up_layers.append(UpBlock(in_dim, out_dim))
        self.decoder = nn.Sequential(*up_layers)

        self.apply(self._init_weights)
        self._fix_init_weight()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _run_blocks(self, feat):
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, h, w)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(b, c, h, w)

    def encode(self, x):
        feat = self.stem(x)
        feat = self._run_blocks(feat)
        moments = self.to_moments(feat)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        feat = self.from_latent(z)
        return self.decoder(feat)

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.sample() if self.sample_posterior else posterior.mode()
        x_hat = self.decode(z)
        return x_hat, posterior
