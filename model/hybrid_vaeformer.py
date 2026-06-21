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
    def __init__(self, in_chans, out_chans, final_block=False):
        super().__init__()
        main_layers = [
            nn.Conv2d(in_chans, out_chans * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        ]
        if not final_block:
            main_layers.extend([nn.BatchNorm2d(out_chans), nn.SiLU(inplace=True)])
        main_layers.append(ConvBNAct(out_chans, out_chans, kernel_size=3, stride=1, act=False))
        self.main = nn.Sequential(*main_layers)
        self.shortcut = ChannelToSpaceShortcut(in_chans, out_chans)
        self.act = nn.Identity() if final_block else nn.SiLU(inplace=True)

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
        up_pairs = list(zip(decode_dims, decode_dims[1:] + [out_chans]))
        self.decoder = nn.Sequential(
            *[
                UpBlock(in_dim, out_dim, final_block=(idx == len(up_pairs) - 1))
                for idx, (in_dim, out_dim) in enumerate(up_pairs)
            ]
        )

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


class StageViT(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        grid_shape,
        window_size=(4, 8),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        learnable_pos=True,
    ):
        super().__init__()
        self.dim = dim
        self.grid_shape = tuple(grid_shape)
        pos_embed = get_2d_sincos_pos_embed(dim, self.grid_shape, cls_token=False)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            requires_grad=learnable_pos,
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    window_size=window_size,
                    window=True,
                    rel_pos_spatial=False,
                    act_layer=QuickGELU,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(dim) if depth > 0 else nn.Identity()
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

    def forward(self, feat):
        if len(self.blocks) == 0:
            return feat
        b, c, h, w = feat.shape
        if (h, w) != self.grid_shape:
            raise ValueError(f"StageViT expected grid {self.grid_shape}, got {(h, w)}.")
        x = feat.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, h, w)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(b, c, h, w)


class HybridHier2VAEformer(nn.Module):
    def __init__(
        self,
        in_chans=69,
        out_chans=69,
        img_size=(128, 256),
        encoder_dims=(256, 512, 768, 1024),
        bottom_latent_dim=34,
        top_latent_dim=34,
        encoder_depths=(1, 2, 2, 4),
        decoder_depths=(4, 2, 2, 1),
        num_heads=(8, 8, 12, 16),
        window_size=((4, 8), (4, 8), (4, 8), (4, 4)),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        sample_posterior=False,
        learnable_pos=True,
        latent_drop_probs=None,
        bottom_drop_period=4,
        bottom_scale_init=0.1,
    ):
        super().__init__()
        if len(encoder_dims) != 4:
            raise ValueError("encoder_dims must have four stages for hier2 model.")
        if len(encoder_depths) != 4 or len(decoder_depths) != 4:
            raise ValueError("encoder_depths and decoder_depths must each have four entries.")
        if len(num_heads) != 4 or len(window_size) != 4:
            raise ValueError("num_heads and window_size must each have four entries.")
        if img_size[0] % 16 != 0 or img_size[1] % 16 != 0:
            raise ValueError(f"img_size {img_size} must be divisible by 16.")

        self.sample_posterior = sample_posterior
        self.latent_drop_probs = latent_drop_probs or {"full": 1.0, "top_only": 0.0}
        self.bottom_drop_period = int(bottom_drop_period) if bottom_drop_period is not None else 0
        self.bottom_latent_dim = bottom_latent_dim
        self.top_latent_dim = top_latent_dim

        c1, c2, c3, c4 = encoder_dims
        h, w = img_size
        grids = [(h // 2, w // 2), (h // 4, w // 4), (h // 8, w // 8), (h // 16, w // 16)]

        self.down1 = DownBlock(in_chans, c1)
        self.enc_stage1 = StageViT(c1, encoder_depths[0], num_heads[0], grids[0], tuple(window_size[0]), mlp_ratio, qkv_bias, drop_path_rate, learnable_pos)
        self.down2 = DownBlock(c1, c2)
        self.enc_stage2 = StageViT(c2, encoder_depths[1], num_heads[1], grids[1], tuple(window_size[1]), mlp_ratio, qkv_bias, drop_path_rate, learnable_pos)
        self.bottom_moments = nn.Conv2d(c2, 2 * bottom_latent_dim, 1)

        self.down3 = DownBlock(c2, c3)
        self.enc_stage3 = StageViT(c3, encoder_depths[2], num_heads[2], grids[2], tuple(window_size[2]), mlp_ratio, qkv_bias, drop_path_rate, learnable_pos)
        self.down4 = DownBlock(c3, c4)
        self.enc_stage4 = StageViT(c4, encoder_depths[3], num_heads[3], grids[3], tuple(window_size[3]), mlp_ratio, qkv_bias, drop_path_rate, learnable_pos)
        self.top_moments = nn.Conv2d(c4, 2 * top_latent_dim, 1)

        self.top_proj = nn.Conv2d(top_latent_dim, c4, 1)
        self.dec_stage4 = StageViT(c4, decoder_depths[0], num_heads[3], grids[3], tuple(window_size[3]), mlp_ratio, qkv_bias, drop_path_rate, learnable_pos)
        self.up4 = UpBlock(c4, c3)
        self.dec_stage3 = StageViT(c3, decoder_depths[1], num_heads[2], grids[2], tuple(window_size[2]), mlp_ratio, qkv_bias, drop_path_rate, learnable_pos)
        self.up3 = UpBlock(c3, c2)

        self.bottom_proj = nn.Conv2d(bottom_latent_dim, c2, 1)
        self.register_buffer("null_bottom", torch.zeros(1, bottom_latent_dim, grids[1][0], grids[1][1]))
        self.bottom_scale = nn.Parameter(torch.tensor(float(bottom_scale_init)))
        self.dec_stage2 = StageViT(c2, decoder_depths[2], num_heads[1], grids[1], tuple(window_size[1]), mlp_ratio, qkv_bias, drop_path_rate, learnable_pos)
        self.up2 = UpBlock(c2, c1)
        self.dec_stage1 = StageViT(c1, decoder_depths[3], num_heads[0], grids[0], tuple(window_size[0]), mlp_ratio, qkv_bias, drop_path_rate, learnable_pos)
        self.up1 = UpBlock(c1, out_chans, final_block=True)

        self.apply(self._init_conv_weights)

    def _init_conv_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def _sample_or_mode(self, posterior):
        return posterior.sample() if self.sample_posterior else posterior.mode()

    def _use_bottom(self, global_step=None):
        if not self.training:
            return True
        if self.bottom_drop_period <= 0 or global_step is None:
            return True
        return int(global_step) % self.bottom_drop_period != 0

    def encode(self, x):
        e1 = self.enc_stage1(self.down1(x))
        e2 = self.enc_stage2(self.down2(e1))
        bottom_posterior = DiagonalGaussianDistribution(self.bottom_moments(e2))
        e3 = self.enc_stage3(self.down3(e2))
        e4 = self.enc_stage4(self.down4(e3))
        top_posterior = DiagonalGaussianDistribution(self.top_moments(e4))
        return {"bottom": bottom_posterior, "top": top_posterior}

    def decode(self, z_top, z_bottom=None):
        b = z_top.shape[0]
        d4 = self.dec_stage4(self.top_proj(z_top))
        d3 = self.dec_stage3(self.up4(d4))
        d2 = self.up3(d3)
        if z_bottom is None:
            z_bottom = self.null_bottom.expand(b, -1, -1, -1)
        d2 = d2 + self.bottom_scale * self.bottom_proj(z_bottom)
        d2 = self.dec_stage2(d2)
        d1 = self.dec_stage1(self.up2(d2))
        return self.up1(d1)

    def forward(self, x, global_step=None):
        posteriors = self.encode(x)
        z_top = self._sample_or_mode(posteriors["top"])
        z_bottom = self._sample_or_mode(posteriors["bottom"])
        if not self._use_bottom(global_step):
            z_bottom = torch.zeros_like(z_bottom)
        x_hat = self.decode(z_top, z_bottom)
        return x_hat, posteriors
