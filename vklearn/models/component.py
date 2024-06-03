from torch import Tensor
import torch.nn as nn


class BasicConvBD(nn.Sequential):

    def __init__(
            self,
            in_planes:   int,
            out_planes:  int,
            kernel_size: int=3,
            stride:      int | tuple[int, int]=1
        ):

        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class LinearBasicConvBD(nn.Sequential):

    def __init__(
            self,
            in_planes:   int,
            out_planes:  int,
            kernel_size: int=3,
            dilation:    int=1,
            stride:      int | tuple[int, int]=1
        ):

        padding = (kernel_size + 2 * (dilation - 1) - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes, in_planes, kernel_size, stride, padding,
                dilation=dilation, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes))


class BasicConvDB(nn.Sequential):

    def __init__(
            self,
            in_planes:   int,
            out_planes:  int,
            kernel_size: int=3,
            stride:      int | tuple[int, int]=1
        ):

        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size, stride, padding, groups=out_planes, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.Hardswish(inplace=True))


class LinearBasicConvDB(nn.Sequential):

    def __init__(
            self,
            in_planes:   int,
            out_planes:  int,
            kernel_size: int=3,
            dilation:    int=1,
            stride:      int | tuple[int, int]=1
        ):

        padding = (kernel_size + 2 * (dilation - 1) - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.Conv2d(out_planes, out_planes, kernel_size, stride, padding,
                dilation=dilation, groups=out_planes, bias=False),
            nn.BatchNorm2d(out_planes))


class UpSample(nn.Sequential):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
        ):

        super().__init__(
            nn.ConvTranspose2d(in_planes, in_planes, 3, 2, 1, output_padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            BasicConvDB(in_planes, out_planes, 3),
        )


class PixelShuffleSample(nn.Sequential):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
        ):

        super().__init__(
            nn.Conv2d(in_planes, in_planes * 2, 1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(in_planes // 2),
            BasicConvBD(in_planes // 2, out_planes, 3),
        )


class CSENet(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            out_planes:    int,
            kernel_size:   int=3,
            shrink_factor: int=4,
        ):

        super().__init__()

        shrink_dim = in_planes // shrink_factor
        padding = (kernel_size - 1) // 2
        self.fusion = nn.Sequential(
            nn.Conv2d(in_planes, shrink_dim, 1, bias=False),
            nn.BatchNorm2d(shrink_dim),
            nn.Conv2d(shrink_dim, shrink_dim, kernel_size, padding=padding, groups=shrink_dim, bias=False),
            nn.BatchNorm2d(shrink_dim),
            nn.Conv2d(shrink_dim, in_planes, 1, bias=False),
            nn.Hardsigmoid(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.project(x * self.fusion(x))
