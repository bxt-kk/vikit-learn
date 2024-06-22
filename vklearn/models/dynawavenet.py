from torch import Tensor
import torch.nn as nn

from .basic import Basic


class DynawaveNet(Basic):
    '''A light-weight and easy-to-train model

    Args:
        num_waves: Number of the global wave blocks.
        wave_depth: Depth of the wave block.
    '''

    def __init__(
            self,
            num_waves:  int=3,
            wave_depth: int=3,
        ):

        super().__init__()

        self.num_waves  = num_waves
        self.wave_depth = wave_depth

        self.features = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(12, 48, 3, padding=1, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, padding=1, stride=2),
            nn.BatchNorm2d(192),
        ) # 192, 32, 32

        self.features_dim = 192

        self.global_waves = nn.ModuleList()
        for _ in range(num_waves):
            global_wave_F = nn.ModuleList()
            global_wave_B = nn.ModuleList()
            for k in range(wave_depth):
                in_channels = self.features_dim * 2**k
                out_channels = in_channels * 2
                global_wave_F.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, groups=self.features_dim),
                    nn.BatchNorm2d(out_channels)))
                global_wave_B.insert(0, nn.Sequential(
                    nn.ConvTranspose2d(out_channels, in_channels, 3, 2, 1, output_padding=1, groups=self.features_dim),
                    nn.BatchNorm2d(in_channels)))
            global_wave_B.insert(0, nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1, groups=self.features_dim),
                nn.BatchNorm2d(out_channels)))

            global_wave_C = nn.Sequential(
                nn.Conv2d(self.features_dim, self.features_dim, 1),
                nn.BatchNorm2d(self.features_dim),
                nn.ReLU(),
                nn.Conv2d(self.features_dim, self.features_dim, 1),
                nn.BatchNorm2d(self.features_dim),
            )
            global_wave_N = nn.BatchNorm2d(self.features_dim)

            self.global_waves.append(nn.ModuleDict(dict(
                wave_F=global_wave_F,
                wave_B=global_wave_B,
                wave_C=global_wave_C,
                wave_N=global_wave_N,
            )))

    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        for global_wave in self.global_waves:
            x0 = x
            vs = [x]
            for wave_f in global_wave['wave_F']:
                x = wave_f(x)
                vs.append(x)
            for wave_b in global_wave['wave_B']:
                x = wave_b(x)
                x = x + vs.pop()
            x = global_wave['wave_N'](x0 + global_wave['wave_C'](x))
        return x
