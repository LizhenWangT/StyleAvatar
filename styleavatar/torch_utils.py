import torch
import random
import math
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torchvision


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    up = (up, up)
    down = (down, down)
    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])
    out = upfirdn2d_native(input, kernel, *up, *down, *pad)
    return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
    return out.view(-1, channel, out_h, out_w)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss 


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed#, embedder_obj.out_dim


def make_noise(channel, res, device):
    device = input.input.device
    noises = torch.randn(1, channel, res, res, device=device)
    return noises


def pad_1_4(tensor, length, value=-1):
    return F.pad(tensor, (length, length, length, length), "constant", value)


def dilate(depth, pix):
    # depth: [B, 1, H,W]
    newdepth = depth.detach().clone()
    for i in range(pix):
        d1 = newdepth[:, :, 1:, :]
        d2 = newdepth[:, :, :-1, :]
        d3 = newdepth[:, :, :, 1:]
        d4 = newdepth[:, :, :, :-1]
        newdepth[:, :, :-1, :] = torch.where(newdepth[:, 0:1, :-1, :] > 0, newdepth[:, :, :-1, :], d1)
        newdepth[:, :, 1:, :] = torch.where(newdepth[:, 0:1, 1:, :] > 0, newdepth[:, :, 1:, :], d2)
        newdepth[:, :, :, :-1] = torch.where(newdepth[:, 0:1, :, :-1] > 0, newdepth[:, :, :, :-1], d3)
        newdepth[:, :, :, 1:] = torch.where(newdepth[:, 0:1, :, 1:] > 0, newdepth[:, :, :, 1:], d4)
        depth = newdepth
    return newdepth


def erode(depth, pix):
    # depth: [B, C, H, W]
    newdepth = depth.detach().clone()
    for i in range(pix):
        d1 = depth[:, :, 1:, :]
        d2 = depth[:, :, :-1, :]
        d3 = depth[:, :, :, 1:]
        d4 = depth[:, :, :, :-1]
        newdepth[:, :, :-1, :] = torch.where(newdepth[:, :, :-1, :] > 0, d1, newdepth[:, :, :-1, :])
        newdepth[:, :, 1:, :] = torch.where(newdepth[:, :, 1:, :] > 0, d2, newdepth[:, :, 1:, :])
        newdepth[:, :, :, :-1] = torch.where(newdepth[:, :, :, :-1] > 0, d3, newdepth[:, :, :, :-1])
        newdepth[:, :, :, 1:] = torch.where(newdepth[:, :, :, 1:] > 0, d4, newdepth[:, :, :, 1:])
        depth = newdepth
    return newdepth


def Neural_Patches(img, patches_size=8, stride=4):
    patches = []
    stride_true = patches_size-stride
    times = img.shape[-1] // stride_true
    if stride_true * times < img.shape[-1]:
        for row in range(times):
            for col in range(times):
                left = col * stride_true
                right = patches_size + col * stride_true
                top = row * stride_true
                down = patches_size + row * stride_true
                if row == times:
                    top = img.shape[-1] - patches_size
                    down = img.shape[-1]
                if col == times:
                    left = img.shape[-1] - patches_size
                    right = img.shape[-1]
                patches.append(img[:, :, top:down, left:right])

    else:
        for row in range(times - 1):
            for col in range(times - 1):
                patches.append(img[:, :, row * stride_true:patches_size + row * stride_true, col * stride_true:patches_size + col * stride_true])
    return torch.cat(patches, dim=0)
