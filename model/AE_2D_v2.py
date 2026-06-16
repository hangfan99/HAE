import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass



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
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean



class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)
    


class Mix_loss(nn.Module):
    def __init__(self,kl_weight=1e-4, KL=True):
        super(Mix_loss, self).__init__()
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)
        self.kl_weight = kl_weight
        self.KL = KL

    def forward(self, x_recon, x, posteriors):
        rec_loss = torch.square(x_recon.contiguous() - x.contiguous())
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if self.KL:
            return nll_loss + self.kl_weight*kl_loss
        else:
            return nll_loss




class AE_KL(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 hidden_dims: List = None,
                 conv_5_depth = 0,
                 out_channels = 1,
                
                 
                 **kwargs) -> None:
        super(AE_KL, self).__init__()

        self.embedding_dim = embedding_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]


        # Build Encoder
        for i,h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                            kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.InstanceNorm2d(h_dim))
            )
            for _ in range(np.min([i*6,24])):
                modules.append(ResidualLayer(h_dim, h_dim))
            in_channels = h_dim
            

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(8):
            modules.append(ResidualLayer(in_channels, in_channels))
            modules.append(nn.LeakyReLU())
        modules.append(torch.nn.Conv2d(in_channels,
                                        2*in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))

        self.encoder = nn.Sequential(*modules)

        self.quant_conv = torch.nn.Conv2d(2*in_channels, 2*embedding_dim, 1)

        self.post_quant_conv = torch.nn.Conv2d(embedding_dim, hidden_dims[-1], 1)

        # Build Decoder
        modules = []
        for _ in range(8):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
            modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            for _ in range(np.min([(len(hidden_dims)-i+1)*6,24])):
                modules.append(ResidualLayer(hidden_dims[i], hidden_dims[i]))
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),
                    nn.LeakyReLU(),
                    nn.InstanceNorm2d(hidden_dims[i + 1])))
            
            
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3*out_channels,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.LeakyReLU(),
                nn.InstanceNorm2d(3*out_channels)))
        
        modules.append(ResidualLayer(3*out_channels, 3*out_channels))
        modules.append(nn.Conv2d(3*out_channels, out_channels, kernel_size=3, stride=1, padding=1))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:

        h = self.encoder(input)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: Tensor) -> Tensor:

        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior