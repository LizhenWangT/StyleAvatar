import math

import torch
from torch import nn

from networks.modules_for_onnx import *
from torch_utils import get_embedder, erode


class DoubleStyleUnet(nn.Module):
    def __init__(self, input_size, output_size, style_dim, mlp_num, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], for_cpp=False):
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
        self.feature_size = self.mid_size // 2
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
            self.cond_convs_0.append(ConvBlock(in_channel, out_channel))
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
            self.convs_0.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs_0.append(StyledConv(out_channel, out_channel, 3, style_dim, upsample=False))
            self.to_rgbs_0.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        
        # final upsample of input 3DMM rendering feature
        self.tex_up = nn.ModuleList()
        self.tex_up.append(StyledConv(self.channels[2 ** self.mid_log_size] // 2, self.channels[2 ** self.mid_log_size] // 2, 3, style_dim, upsample=True))
        self.tex_up.append(StyledConv(self.channels[2 ** self.mid_log_size] // 2, self.channels[2 ** self.mid_log_size], 3, style_dim, activation=False))
        self.tex_up.append(ToRGB(self.channels[2 ** self.mid_log_size], style_dim))
        self.get_mask = ConvLayer(self.channels[2 ** self.mid_log_size] * 2, 1, 3, activate=False)
        self.cond_addition = ConvBlock(self.channels[2 ** self.mid_log_size], self.channels[2 ** (self.mid_log_size + 1)])
        
        # for the final image generation
        self.from_rgbs_1 = nn.ModuleList()
        self.cond_convs_1 = nn.ModuleList()
        self.comb_convs_1 = nn.ModuleList()

        in_channel = self.channels[self.input_size] * 2
        for i in range(self.mid_log_size - 2, 2, -1):
            out_channel = self.channels[2 ** i]
            self.from_rgbs_1.append(FromRGB(in_channel, 3, downsample=True))
            self.cond_convs_1.append(ConvBlock(in_channel, out_channel))
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
            self.convs_1.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs_1.append(StyledConv(out_channel, out_channel, 3, style_dim))
            self.to_rgbs_1.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        
        self.noise_face = torch.randn(1, 21, self.feature_size, self.feature_size).cuda()
        self.noise_back = torch.randn(1, 21, self.feature_size, self.feature_size).cuda()
        self.noise_base = torch.randn(1, 21, self.mid_size, self.mid_size).cuda()
        embedder = get_embedder(32)
        frame_num = torch.zeros([1, 1]).cuda() + 0.5
        self.style_in = embedder(frame_num)[:, :style_dim].cuda()

        tmp_channel = self.channels[self.feature_size]
        self.feature_back = torch.randn(1, tmp_channel, self.feature_size, self.feature_size).cuda()
        self.skip_back = torch.randn(1, 12, self.feature_size, self.feature_size).cuda()
        self.feature_face = torch.randn(1, tmp_channel, self.feature_size, self.feature_size).cuda()
        self.skip_face = torch.randn(1, 12, self.feature_size, self.feature_size).cuda()

        new_h = torch.linspace(-0.5, 0.5, self.feature_size).view(-1, 1).repeat(1, self.feature_size)
        new_w = torch.linspace(-0.5, 0.5, self.feature_size).repeat(self.feature_size, 1)
        self.uv_position = torch.cat([new_w.unsqueeze(2), new_h.unsqueeze(2)], dim=2).unsqueeze(0).cuda()

    def assigen(self, tensor_ckpt):
        left = self.mid_size // 4
        top = self.mid_size // 4
        self.feature_face = tensor_ckpt['feature_face'][0:1].cuda()
        self.skip_face = tensor_ckpt['skip_face'][0:1].cuda()
        self.feature_back = F.upsample(tensor_ckpt['feature_back'][0:1].cuda(), size=(self.mid_size, self.mid_size), mode='bilinear')[:, :, top:top + self.feature_size, left:left + self.feature_size]
        self.skip_back = F.upsample(tensor_ckpt['skip_back'][0:1].cuda(), size=(self.mid_size, self.mid_size), mode='bilinear')[:, :, top:top + self.feature_size, left:left + self.feature_size]

    def forward(self, condition_img, uv_image):
        latent = self.mapping(self.pixelnorm(self.style_in))

        if self.for_cpp:
            condition_img = torch.flip(condition_img, dims=[3])
            condition_img = condition_img.permute(0, 3, 1, 2) / 127.5 - 1
            uv_image = torch.flip(uv_image, dims=[3])
            uv_image = uv_image.permute(0, 3, 1, 2) / 127.5 - 1
        
        mask_face = (uv_image[:, 2:3] > -0.999).type(torch.float32)
        mask_face = erode(mask_face, 1)
        # face feature warping
        feature_warp = F.grid_sample(self.feature_face, uv_image[:, :2].permute(0, 2, 3, 1), align_corners=False, mode='nearest') * mask_face
        skip_warp = F.grid_sample(self.skip_face, uv_image[:, :2].permute(0, 2, 3, 1), align_corners=False, mode='nearest') * mask_face
        noise_warp = F.grid_sample(self.noise_face, uv_image[:, :2].permute(0, 2, 3, 1), align_corners=False, mode='nearest') * mask_face

        cond_img = condition_img#F.pad(condition_img, (self.input_size // 4, self.input_size // 4, self.input_size // 4, self.input_size // 4), "constant", -1)

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
        feature_base = F.grid_sample(feature_tex, self.uv_position, align_corners=False, mode='nearest')
        noises_base = F.grid_sample(self.noise_base, self.uv_position, align_corners=False, mode='nearest')
        skip_base = F.grid_sample(skip, self.uv_position, align_corners=False, mode='nearest')

        # foreground mask prediction (use grad or not & use gt mask or predicted mask)
        out_mask = torch.tanh(self.get_mask(torch.cat([feature_base, self.feature_back], dim=1)))
        out_mask_process = torch.clip(out_mask * 1.1, -1, 1) / 2 + 0.5
        
        # feature fusion, process noises for less "texture skinning"
        feature_fused = feature_base * (1 - out_mask_process) + self.feature_back * out_mask_process + feature_warp
        noises_fused = noises_base * (1 - out_mask_process) + self.noise_back * out_mask_process + noise_warp
        skip_fused = skip_base * (1 - out_mask_process) + self.skip_back * out_mask_process + skip_warp
        noise_6 = noises_fused[:, :1]
        noise_8 = self.iwt_1(noises_fused[:, 1:5])
        noise_10 = self.iwt_1(self.iwt_4(noises_fused[:, 5:21]))
        
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

        if self.for_cpp:
            image = torch.clamp(image.permute(0, 2, 3, 1) * 127.5 + 127.5, 0, 255)
            image = torch.flip(image, dims=[3])

        return image
