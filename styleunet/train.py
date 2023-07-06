import argparse
import os

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils, models
from tqdm import tqdm

from networks.generator import StyleUNet
from networks.discriminator import Discriminator
from distributed import (
    get_rank,
    synchronize
)
from augmentation import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
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


def train(args, loader, generator, discriminator, g_ema, g_optim, d_optim, device):
    torch.manual_seed(0)
    loader = sample_data(loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    vgg = VGGLoss(device)

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    if args.start_iter != 0:
        for p in g_optim.param_groups:
            p['lr'] = args.lr * min(1, 5000 / args.start_iter)
        for p in d_optim.param_groups:
            p['lr'] = args.lr * min(1, 5000 / args.start_iter)
    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img, cond_img = next(loader)
        real_img = real_img.to(device)
        cond_img = cond_img.to(device)
        latent = torch.randn(args.batch, args.style_dim).to(device)
        if args.input_size != args.output_size:
            cond_img_resize = F.interpolate(cond_img, size=(args.output_size, args.output_size), mode='bilinear', align_corners=False)
        else:
            cond_img_resize = cond_img
        
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        if args.augment:
            cond_img, aug_mat = augment(cond_img, ada_aug_p, (None, None), use_affine=False, use_color=True)
            real_img, _ = augment(real_img, ada_aug_p, aug_mat, use_affine=False, use_color=True)

        fake_img = generator(cond_img, latent)

        if args.use_concat:
            fake_pred = discriminator(torch.cat([fake_img, cond_img_resize], dim=1))
            real_pred = discriminator(torch.cat([real_img, cond_img_resize], dim=1))
        else:
            fake_pred = discriminator(fake_img)
            real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            cond_img_resize.requires_grad = True
            cond_img.requires_grad = True

            if args.use_concat:
                real_pred = discriminator(torch.cat([real_img, cond_img_resize], dim=1))
            else:
                real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_img = generator(cond_img, latent)

        if args.use_concat:
            fake_pred = discriminator(torch.cat([fake_img, cond_img_resize], dim=1))
        else:
            fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred) * args.g_loss_w
        l1_loss = torch.mean(torch.abs(fake_img - real_img)) * args.l1_loss_w
        vgg_loss = vgg(F.upsample(fake_img, size=(512, 512), mode='bilinear'), F.upsample(real_img, size=(512, 512), mode='bilinear')) * args.vgg_loss_w

        generator.zero_grad()
        (g_loss + l1_loss + vgg_loss).backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        d_loss_val = d_loss.mean().item()
        g_loss_val = g_loss.mean().item()
        l1_loss_val = l1_loss.mean().item()
        vgg_loss_val = vgg_loss.mean().item()

        if get_rank() == 0:
            pbar.set_description((f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; l1: {l1_loss_val:.4f}; vgg: {vgg_loss_val:.4f} "))

            if i % 1000 == 0:# and i != args.start_iter:
                with torch.no_grad():
                    sample = g_ema(cond_img, latent).clamp(min=-1, max=1)
                    utils.save_image(
                        torch.cat([real_img, sample, cond_img_resize], dim=3),
                        f"logs/sample/{args.savename}_{str(i).zfill(6)}.png",
                        nrow=int(args.batch ** 0.5),
                        normalize=True,
                        range=(-1, 1)
                    )

            if i % 5000 == 0 and i != args.start_iter:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args
                    },
                    f"logs/checkpoint/{args.savename}_{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    os.makedirs("logs/checkpoint", exist_ok=True)
    os.makedirs("logs/sample", exist_ok=True)
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=4, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=1024, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation")
    parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_length", type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_every", type=int, default=256, help="probability update interval of the adaptive augmentation")
    parser.add_argument("--mode", type=int, default=0, help="0 for inpainting; 1 for superresolution; 2 for retouching; 3 for 3dmm")
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    args.g_loss_w = 1
    args.l1_loss_w = 5
    args.vgg_loss_w = 0.03
    args.use_concat = False
    if args.mode == 0:
        from dataset import InpaitingDataset
        dataset = InpaitingDataset(args.path, transform)
        args.savename = 'inpainting'
        args.input_size = 1024
        args.output_size = 1024
    elif args.mode == 1:
        from dataset import SuperResolutionDataset
        dataset = SuperResolutionDataset(args.path, transform)
        args.savename = 'superresolution'
        args.input_size = 1024
        args.output_size = 1024
    elif args.mode == 2:
        from dataset import RetouchDataset
        dataset = RetouchDataset(args.path, transform)
        args.savename = 'retouching'
        args.input_size = 1024
        args.output_size = 1024
    elif args.mode == 3:
        from dataset import TDMMDataset
        dataset = TDMMDataset(args.path, transform)
        args.savename = 'tdmm'
        args.input_size = 256
        args.output_size = 1024
        args.g_loss_w = 1
        args.l1_loss_w = 1
        args.vgg_loss_w = 0
        args.use_concat = True

    device = "cuda"
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.n_mlp = 4
    args.start_iter = 0
    args.style_dim = 64
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    generator = StyleUNet(args.input_size, args.output_size, args.style_dim, args.n_mlp, args.channel_multiplier).to(device)
    if args.use_concat:
        discriminator = Discriminator(args.output_size, 6).to(device)
    else:
        discriminator = Discriminator(args.output_size, 3).to(device)
    g_ema = StyleUNet(args.input_size, args.output_size, args.style_dim, args.n_mlp, args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0].split('_')[-1])
        except ValueError:
            pass
        generator.load_state_dict(ckpt["g"], strict=False)
        discriminator.load_state_dict(ckpt["d"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

    loader = data.DataLoader(dataset, batch_size=args.batch, sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed), drop_last=True)

    train(args, loader, generator, discriminator, g_ema, g_optim, d_optim, device)
