import argparse
import os

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils, models
from tqdm import tqdm
import numpy as np
import torch_utils

from networks.generator import DoubleStyleUnet, FeatureGenerator
from networks.discriminator import Discriminator
from distributed import (
    get_rank,
    synchronize
)


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


def train(args, loader, back_generator, face_generator, image_generator, discriminator, g_ema, b_g_optim, f_g_optim, i_g_optim, d_optim, device):
    torch.manual_seed(0)
    loader = sample_data(loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    vgg = VGGLoss(device)

    if args.distributed:
        b_module = back_generator.module
        f_module = face_generator.module
        i_module = image_generator.module
        d_module = discriminator.module
    else:
        b_module = back_generator
        f_module = face_generator
        i_module = image_generator
        d_module = discriminator

    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    if args.start_iter != 0:
        for p in b_g_optim.param_groups:
            p['lr'] = args.lr * min(1, 5000 / args.start_iter)
        for p in f_g_optim.param_groups:
            p['lr'] = args.lr * min(1, 5000 / args.start_iter)
        for p in i_g_optim.param_groups:
            p['lr'] = args.lr * min(1, 5000 / args.start_iter)
        for p in d_optim.param_groups:
            p['lr'] = args.lr * min(1, 5000 / args.start_iter)
    
    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        if idx % 2 == 0:
            dis_x = np.clip(np.random.randn(1) * args.feature_size // 12 + args.feature_size // 4, 10, args.feature_size // 2 - 10).astype(np.int32)[0]
            dis_y = np.clip(np.random.randn(1) * args.feature_size // 12 + args.feature_size // 4, 10, args.feature_size // 2 - 10).astype(np.int32)[0]
        else:
            dis_x = args.feature_size // 4
            dis_y = args.feature_size // 4
        
        left = dis_x / args.feature_size - 0.75
        right = dis_x / args.feature_size + 0.25
        top = dis_y / args.feature_size - 0.75
        down = dis_y / args.feature_size + 0.25
        new_h = torch.linspace(top, down, args.feature_size).view(-1, 1).repeat(1, args.feature_size)
        new_w = torch.linspace(left, right, args.feature_size).repeat(args.feature_size, 1)
        uv_position = torch.cat([new_w.unsqueeze(2), new_h.unsqueeze(2)], dim=2).unsqueeze(0).to(device).repeat(args.batch, 1, 1, 1)

        image, uv, render, back, frame_latent, video_latent = next(loader)
        image = image[:, :, dis_y * 8:dis_y * 8 + args.feature_size * 8, dis_x * 8:dis_x * 8 + args.feature_size * 8].to(device)
        back = back[:, :, dis_y * 8:dis_y * 8 + args.feature_size * 8, dis_x * 8:dis_x * 8 + args.feature_size * 8].to(device)
        uv = torch_utils.pad_1_4(uv, 64)[:, :, dis_y * 2:dis_y * 2 + args.feature_size * 2, dis_x * 2:dis_x * 2 + args.feature_size * 2].to(device)
        render = render.to(device)
        frame_latent = frame_latent[:, :args.latent_frame].to(device)
        video_latent = video_latent[:, :args.latent_video].to(device)
        
        image_512 = F.upsample(image, size=(512, 512), mode='nearest')
        render_dis = torch_utils.pad_1_4(render, 64)[:, :, dis_y * 2:dis_y * 2 + args.feature_size * 2, dis_x * 2:dis_x * 2 + args.feature_size * 2]
        render_for_concat = F.upsample(render_dis, size=(args.output_size, args.output_size), mode='nearest')
        uv_img_half = F.upsample(uv, scale_factor=0.5, mode='nearest')
        mask = F.upsample(1 - (back[:, 2:3] > 0).type(torch.float32), size=(args.feature_size, args.feature_size), mode='nearest')
        mask_face = (uv_img_half[:, 2:3] > -0.999).type(torch.float32)
        mask_face = torch_utils.erode(mask_face, 1)

        noise_base = torch.randn(args.batch, 21, args.feature_size * 2, args.feature_size * 2, device=device)
        noise_back = torch.randn(args.batch, 21, args.feature_size, args.feature_size, device=device)
        noise_face = torch.randn(args.batch, 21, args.feature_size, args.feature_size, device=device)

        requires_grad(back_generator, True)
        requires_grad(face_generator, True)
        requires_grad(image_generator, True)
        requires_grad(discriminator, False)

        feature_back, skip_back = back_generator(video_latent)
        side_size = args.feature_size // 4
        feature_back = F.upsample(feature_back, size=(args.feature_size * 2, args.feature_size * 2), mode='bilinear')
        feature_back = feature_back[:, :, dis_y + side_size:dis_y + side_size + args.feature_size, dis_x + side_size:dis_x + side_size + args.feature_size]
        skip_back = F.upsample(skip_back, size=(args.feature_size * 2, args.feature_size * 2), mode='bilinear')
        skip_back = skip_back[:, :, dis_y + side_size:dis_y + side_size + args.feature_size, dis_x + side_size:dis_x + side_size + args.feature_size]
        feature_face, skip_face = face_generator(video_latent)

        fake_img, out_mask = image_generator(frame_latent, render, uv_position, feature_face, feature_back, skip_face, skip_back, uv_img_half, 
                                            mask_face, noise_base, noise_face, noise_back, mask_gt=None, stop_mask_grad=0)
        out_mask = out_mask / 2 + 0.5
        mask_loss = torch.mean(torch.abs(mask - out_mask)) * 10 #* (1 - stop_mask_grad_)
        l1_loss = torch.mean(torch.abs(fake_img - image)) * 5
        vgg_loss = vgg(F.upsample(fake_img, size=(512, 512), mode='bilinear'), image_512)

        if args.use_concat:
            fake_pred = discriminator(torch.cat([fake_img, render_for_concat], dim=1))
            real_pred = discriminator(torch.cat([image, render_for_concat], dim=1))
        else:
            fake_pred = discriminator(fake_img)
            real_pred = discriminator(image)
        g_loss = g_nonsaturating_loss(fake_pred)

        back_generator.zero_grad()
        face_generator.zero_grad()
        image_generator.zero_grad()
        (l1_loss + vgg_loss + g_loss + mask_loss).backward()
        nn.utils.clip_grad_norm_(back_generator.parameters(), max_norm=1, norm_type=2)
        nn.utils.clip_grad_norm_(face_generator.parameters(), max_norm=1, norm_type=2)
        nn.utils.clip_grad_norm_(image_generator.parameters(), max_norm=1, norm_type=2)
        b_g_optim.step()
        f_g_optim.step()
        i_g_optim.step()

        requires_grad(back_generator, False)
        requires_grad(face_generator, False)
        requires_grad(image_generator, False)
        requires_grad(discriminator, True)

        if args.use_concat:
            fake_pred = discriminator(torch.cat([fake_img.detach(), render_for_concat], dim=1))
            real_pred = discriminator(torch.cat([image, render_for_concat], dim=1))
        else:
            fake_pred = discriminator(fake_img.detach())
            real_pred = discriminator(image)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if i % args.d_reg_every == 0:
            image.requires_grad = True
            render_for_concat.requires_grad = True
            real_pred = discriminator(torch.cat([image, render_for_concat], dim=1))
            r1_loss = d_r1_loss(real_pred, image)
            discriminator.zero_grad()
            (5 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        accumulate(g_ema, i_module, accum)

        d_loss_val = d_loss.mean().item()
        g_loss_val = g_loss.mean().item()
        l1_loss_val = l1_loss.mean().item()
        mask_loss_val = mask_loss.mean().item()
        vgg_loss_val = vgg_loss.mean().item()

        if get_rank() == 0:
            pbar.set_description((f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; l1: {l1_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; mask: {mask_loss_val:.4f} "))

            if i % 1000 == 0:# and i != args.start_iter:
                with torch.no_grad():
                    sample, _ = g_ema(frame_latent, render, uv_position, feature_face, feature_back, skip_face, skip_back, uv_img_half, 
                                            mask_face, noise_base, noise_face, noise_back, mask_gt=None, stop_mask_grad=0)
                    out_mask = F.upsample(out_mask, size=(1024, 1024), mode='bilinear').repeat(1, 3, 1, 1)
                    mask = F.upsample(mask, size=(1024, 1024), mode='bilinear').repeat(1, 3, 1, 1)
                    utils.save_image(
                        torch.cat([image, sample.clamp(min=-1, max=1), render_for_concat, out_mask.clamp(min=-1, max=1), mask], dim=3),
                        f"logs/sample/{args.savename}_{str(i).zfill(6)}.png",
                        nrow=int(args.batch ** 0.5),
                        normalize=True,
                        range=(-1, 1)
                    )

            if i % 5000 == 0 and i != args.start_iter:
                with torch.no_grad():
                    feature_back, skip_back = back_generator(video_latent)
                    feature_face, skip_face = face_generator(video_latent)
                torch.save(
                    {
                        "b_g": b_module.state_dict(),
                        "f_g": f_module.state_dict(),
                        "i_g": i_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "b_g_optim": b_g_optim.state_dict(),
                        "f_g_optim": f_g_optim.state_dict(),
                        "i_g_optim": i_g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "feature_back": feature_back.detach().clone(),
                        "skip_back": skip_back.detach().clone(),
                        "feature_face": feature_face.detach().clone(),
                        "skip_face": skip_face.detach().clone()
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
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    args.g_loss_w = 1
    args.l1_loss_w = 5
    args.vgg_loss_w = 0.03
    args.use_concat = True

    from dataset import TrainDataset
    dataset = TrainDataset(args.path, transform)
    args.savename = 'tdmm'
    try:
        args.savename = os.path.basename(args.path)
    except ValueError:
        pass
    
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
    args.input_size = 256
    args.output_size = 1024
    args.feature_size = args.input_size // 2
    args.latent_video = 64
    args.latent_frame = 64
    args.n_mlp = 4

    back_generator = FeatureGenerator(args.feature_size, args.latent_video, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    face_generator = FeatureGenerator(args.feature_size, args.latent_video, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    image_generator = DoubleStyleUnet(args.input_size, args.output_size, args.latent_frame, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)

    if args.use_concat:
        discriminator = Discriminator(args.output_size, 6).to(device)
    else:
        discriminator = Discriminator(args.output_size, 3).to(device)
    g_ema = DoubleStyleUnet(args.input_size, args.output_size, args.latent_frame, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, image_generator, 0)

    b_g_optim = optim.AdamW(back_generator.parameters(), lr=args.lr, betas=(0, 0.99))
    f_g_optim = optim.AdamW(face_generator.parameters(), lr=args.lr, betas=(0, 0.99))
    i_g_optim = optim.AdamW(image_generator.parameters(), lr=args.lr, betas=(0, 0.99))
    d_optim = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0, 0.99))

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0].split('_')[-1])
            #args.start_iter = 4000
        except ValueError:
            pass

        back_generator.load_state_dict(ckpt["b_g"], strict=False)
        face_generator.load_state_dict(ckpt["f_g"], strict=False)
        image_generator.load_state_dict(ckpt["i_g"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        b_g_optim.load_state_dict(ckpt["b_g_optim"])
        f_g_optim.load_state_dict(ckpt["f_g_optim"])
        i_g_optim.load_state_dict(ckpt["i_g_optim"])

        discriminator.load_state_dict(ckpt["d"], strict=False)
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        back_generator = nn.parallel.DistributedDataParallel(back_generator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        face_generator = nn.parallel.DistributedDataParallel(face_generator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        image_generator = nn.parallel.DistributedDataParallel(image_generator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

    loader = data.DataLoader(dataset, batch_size=args.batch, sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed), drop_last=True)

    train(args, loader, back_generator, face_generator, image_generator, discriminator, g_ema, b_g_optim, f_g_optim, i_g_optim, d_optim, device)
