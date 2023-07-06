import math

import torch
from torch import nn

from networks.modules_for_onnx import *


class StyleUNet(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        style_dim=64,
        mlp_num=4,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        device="cpu",
        for_cpp=False
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.style_dim = style_dim
        self.mlp_num = mlp_num
        self.for_cpp = for_cpp

        self.pixelnorm = PixelNorm()
        mlp = []
        for i in range(mlp_num):
            mlp.append(nn.Linear(style_dim, style_dim, bias=True))
            if i < mlp_num - 1:
                mlp.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.mapping = nn.Sequential(*mlp)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.in_log_size = int(math.log(input_size, 2)) - 1
        self.out_log_size = int(math.log(output_size, 2)) - 1
        self.comb_num = self.in_log_size - 5

        # add new layer here
        self.dwt = HaarTransform(3)
        self.from_rgbs = nn.ModuleList()
        self.cond_convs = nn.ModuleList()
        self.comb_convs = nn.ModuleList()

        in_channel = self.channels[self.input_size]
        for i in range(self.in_log_size - 2, 2, -1):
            out_channel = self.channels[2 ** i]
            self.from_rgbs.append(FromRGB(in_channel, 3, downsample=True))
            self.cond_convs.append(ConvBlock(in_channel, out_channel, blur_kernel))
            if i > 3:
                self.comb_convs.append(ConvLayer(out_channel * 2, out_channel, 3))
            else:
                self.comb_convs.append(ConvLayer(out_channel, out_channel, 3))
            in_channel = out_channel

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[8]

        for i in range(4, self.out_log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, upsample=False))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        self.iwt = InverseHaarTransform(3)

        self.noises = []
        for i in range(4, self.out_log_size + 1):
            for _ in range(2):
                self.noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

    def forward(self, condition_img, style):
        latent = self.mapping(self.pixelnorm(style))

        if self.for_cpp:
            condition_img = torch.flip(condition_img, dims=[3])
            condition_img = condition_img.permute(0, 3, 1, 2) / 127.5 - 1

        cond_img = self.dwt(condition_img)
        cond_out = None
        cond_list = []
        for from_rgb, cond_conv in zip(self.from_rgbs, self.cond_convs):
            cond_img, cond_out = from_rgb(cond_img, cond_out)
            cond_out = cond_conv(cond_out)
            cond_list.append(cond_out)

        i = 0
        skip = None
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            if i == 0:
                out = self.comb_convs[self.comb_num](cond_list[self.comb_num])
            elif i < self.comb_num * 2 + 1:
                out = torch.cat([out, cond_list[self.comb_num - (i//2)]], dim=1)
                out = self.comb_convs[self.comb_num - (i//2)](out)
            out = conv1(out, latent, self.noises[i])
            out = conv2(out, latent, self.noises[i + 1])
            skip = to_rgb(out, latent, skip)

            i += 2

        image = self.iwt(skip)

        if self.for_cpp:
            image = torch.clamp(image.permute(0, 2, 3, 1) * 127.5 + 127.5, 0, 255)
            image = torch.flip(image, dims=[3])

        return image
