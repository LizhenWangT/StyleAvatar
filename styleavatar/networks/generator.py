import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from networks.modules import *


class FeatureGenerator(nn.Module):
    def __init__(self, output_size, style_dim, mlp_num, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

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

        self.output_size = output_size
        self.style_dim = style_dim
        self.mlp_num = mlp_num

        self.pixelnorm = PixelNorm()
        mlp = []
        for i in range(mlp_num):
            mlp.append(nn.Linear(style_dim, style_dim, bias=True))
            if i < mlp_num - 1:
                mlp.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.mapping = nn.Sequential(*mlp)

        self.out_log_size = int(math.log(output_size, 2))

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        self.input = ConstantInput(self.channels[4])

        in_channel = self.channels[4]
        for i in range(3, self.out_log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, upsample=False))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        
        self.batchnorm = nn.BatchNorm2d(in_channel)

    def forward(self, style_in):
        latent = self.mapping(self.pixelnorm(style_in))

        out = self.input(latent)
        
        i = 0
        skip = None
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            out = conv1(out, latent)
            out = conv2(out, latent)
            skip = to_rgb(out, latent, skip)
            i += 2
        out = self.batchnorm(out)

        return out, skip


class DoubleStyleUnet(nn.Module):
    def __init__(self, input_size, output_size, style_dim, mlp_num, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

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

        self.input_size = input_size * 2
        self.mid_size = input_size
        self.output_size = output_size
        self.style_dim = style_dim
        self.mlp_num = mlp_num

        self.pixelnorm = PixelNorm()
        mlp = []
        for i in range(mlp_num):
            mlp.append(nn.Linear(style_dim, style_dim, bias=True))
            if i < mlp_num - 1:
                mlp.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.mapping = nn.Sequential(*mlp)

        self.in_log_size = int(math.log(self.input_size, 2)) - 1
        self.mid_log_size = int(math.log(self.mid_size, 2)) - 1
        self.out_log_size = int(math.log(self.output_size, 2)) - 1
        self.comb_num_0 = self.in_log_size - 7
        self.comb_num_1 = self.mid_log_size - 5

        self.dwt = HaarTransform(3)
        self.iwt = InverseHaarTransform(3)
        self.iwt_1 = InverseHaarTransform(1)
        self.iwt_4 = InverseHaarTransform(4)

        # for the input 3DMM rendering
        self.from_rgbs_0 = nn.ModuleList()
        self.cond_convs_0 = nn.ModuleList()
        self.comb_convs_0 = nn.ModuleList()

        in_channel = self.channels[self.input_size] // 2
        for i in range(self.in_log_size - 2, 3, -1):
            out_channel = self.channels[2 ** i] // 2
            self.from_rgbs_0.append(FromRGB(in_channel, 3, downsample=True))
            self.cond_convs_0.append(ConvBlock(in_channel, out_channel, blur_kernel))
            if i > 4 and i < self.in_log_size - 2:
                self.comb_convs_0.append(ConvLayer(out_channel * 2, out_channel, 3))
            elif i <= 4:
                self.comb_convs_0.append(ConvLayer(out_channel, out_channel, 3))
            in_channel = out_channel

        self.convs_0 = nn.ModuleList()
        self.to_rgbs_0 = nn.ModuleList()

        in_channel = self.channels[16] // 2

        for i in range(5, self.mid_log_size + 1):
            out_channel = self.channels[2 ** i] // 2
            self.convs_0.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs_0.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs_0.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        
        # final upsample of input 3DMM rendering feature
        self.tex_up = nn.ModuleList()
        self.tex_up.append(StyledConv(self.channels[2 ** self.mid_log_size] // 2, self.channels[2 ** self.mid_log_size] // 2, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
        self.tex_up.append(StyledConv(self.channels[2 ** self.mid_log_size] // 2, self.channels[2 ** self.mid_log_size], 3, style_dim, blur_kernel=blur_kernel, activation=False))
        self.tex_up.append(ToRGB(self.channels[2 ** self.mid_log_size], style_dim))
        self.get_mask = ConvLayer(self.channels[2 ** self.mid_log_size] * 2, 1, 3, activate=False)
        self.cond_addition = ConvBlock(self.channels[2 ** self.mid_log_size], self.channels[2 ** (self.mid_log_size + 1)], blur_kernel)
        
        # for the final image generation
        self.from_rgbs_1 = nn.ModuleList()
        self.cond_convs_1 = nn.ModuleList()
        self.comb_convs_1 = nn.ModuleList()

        in_channel = self.channels[self.input_size] * 2
        for i in range(self.mid_log_size - 2, 2, -1):
            out_channel = self.channels[2 ** i]
            self.from_rgbs_1.append(FromRGB(in_channel, 3, downsample=True))
            self.cond_convs_1.append(ConvBlock(in_channel, out_channel, blur_kernel))
            if i > 3:
                self.comb_convs_1.append(ConvLayer(out_channel * 2, out_channel, 3))
            else:
                self.comb_convs_1.append(ConvLayer(out_channel, out_channel, 3))
            in_channel = out_channel

        self.convs_1 = nn.ModuleList()
        self.to_rgbs_1 = nn.ModuleList()

        in_channel = self.channels[8]
        for i in range(4, self.out_log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs_1.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs_1.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs_1.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel


    def forward(self, style_in, condition_img, uv_position, feature_face, feature_back, skip_face, skip_back, uv_image, mask_face, noise_base, noise_face, noise_back, mask_gt=None, stop_mask_grad=0):
        # face feature warping
        feature_warp = F.grid_sample(feature_face, uv_image[:, :2].permute(0, 2, 3, 1), align_corners=False, mode='nearest') * mask_face
        skip_warp = F.grid_sample(skip_face, uv_image[:, :2].permute(0, 2, 3, 1), align_corners=False, mode='nearest') * mask_face
        noise_warp = F.grid_sample(noise_face, uv_image[:, :2].permute(0, 2, 3, 1), align_corners=False, mode='nearest') * mask_face

        cond_img = F.pad(condition_img, (self.input_size // 4, self.input_size // 4, self.input_size // 4, self.input_size // 4), "constant", -1)

        latent = self.mapping(self.pixelnorm(style_in))

        # 3dmm texture feature encode
        cond_img = self.dwt(cond_img)
        cond_out = None
        cond_list_0 = []
        cond_num = 0
        for from_rgb, cond_conv in zip(self.from_rgbs_0, self.cond_convs_0):
            cond_img, cond_out = from_rgb(cond_img, cond_out)
            cond_out = cond_conv(cond_out)
            if cond_num > 0:
                cond_list_0.append(cond_out)
            cond_num += 1
        
        # 3dmm texture feature decode
        i = 0
        skip = None
        for conv1, conv2, to_rgb in zip(self.convs_0[::2], self.convs_0[1::2], self.to_rgbs_0):
            if i == 0:
                out = self.comb_convs_0[self.comb_num_0](cond_list_0[self.comb_num_0])
            elif i < self.comb_num_0 * 2 + 1:
                out = torch.cat([out, cond_list_0[self.comb_num_0 - (i//2)]], dim=1)
                out = self.comb_convs_0[self.comb_num_0 - (i//2)](out)
            out = conv1(out, latent)
            out = conv2(out, latent)
            skip = to_rgb(out, latent, skip)
            i += 2
        
        # 3dmm texture feature upsample
        feature_tex = self.tex_up[0](out, latent)
        feature_tex = self.tex_up[1](feature_tex, latent)
        skip = self.tex_up[2](feature_tex, latent, skip)
        
        # feature translation for training
        feature_base = F.grid_sample(feature_tex, uv_position, align_corners=False, mode='nearest')
        noises_base = F.grid_sample(noise_base, uv_position, align_corners=False, mode='nearest')
        skip_base = F.grid_sample(skip, uv_position, align_corners=False, mode='nearest')

        # foreground mask prediction (use grad or not & use gt mask or predicted mask)
        if stop_mask_grad == 0:
            out_mask = torch.tanh(self.get_mask(torch.cat([feature_base, feature_back.detach()], dim=1)))
        else:
            out_mask = torch.tanh(self.get_mask(torch.cat([feature_base.detach(), feature_back.detach()], dim=1)))
        if mask_gt is not None:
            out_mask_process = torch.clip(mask_gt.detach() * 1.1, -1, 1) / 2 + 0.5
        else:
            out_mask_process = torch.clip(out_mask.detach() * 1.1, -1, 1) / 2 + 0.5
        
        # feature fusion, process noises for less "texture skinning"
        feature_fused = feature_base * (1 - out_mask_process) + feature_back * out_mask_process + feature_warp
        noises_fused = noises_base * (1 - out_mask_process) + noise_back * out_mask_process + noise_warp
        skip_fused = skip_base * (1 - out_mask_process) + skip_back * out_mask_process + skip_warp
        noise_6 = noises_fused[:, :1].contiguous()
        noise_8 = self.iwt_1(noises_fused[:, 1:5], contiguous=True)
        noise_10 = self.iwt_1(self.iwt_4(noises_fused[:, 5:21], contiguous=True), contiguous=True)
        #out_mask = feature_fused[:, :1]

        # image generation encode
        cond_img = skip_fused
        cond_out = self.cond_addition(feature_fused)
        cond_list_1 = []
        cond_num = 0
        for from_rgb, cond_conv in zip(self.from_rgbs_1, self.cond_convs_1):
            cond_img, cond_out = from_rgb(cond_img, cond_out)
            cond_out = cond_conv(cond_out)
            cond_list_1.append(cond_out)
            cond_num += 1

        # image generation decode
        i = 0
        skip = None
        for conv1, conv2, to_rgb in zip(self.convs_1[::2], self.convs_1[1::2], self.to_rgbs_1):
            noise_this = None
            if i == 6:
                noise_this = noise_6
            elif i == 8:
                noise_this = noise_8
            elif i == 10:
                noise_this = noise_10
            if i == 0:
                out = self.comb_convs_1[self.comb_num_1](cond_list_1[self.comb_num_1])
            elif i < self.comb_num_1 * 2 + 1:
                out = torch.cat([out, cond_list_1[self.comb_num_1 - (i//2)]], dim=1)
                out = self.comb_convs_1[self.comb_num_1 - (i//2)](out)
            out = conv1(out, latent, noise=noise_this)
            out = conv2(out, latent, noise=noise_this)
            skip = to_rgb(out, latent, skip)
            i += 2
        
        image = self.iwt(skip)

        return image, out_mask

