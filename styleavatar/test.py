import os
import numpy as np
import argparse
import cv2

import torch
from torch.nn import functional as F

from networks.generator_for_onnx import DoubleStyleUnet
from torchvision import transforms, utils
from PIL import Image



def test(args, generator, device):
    os.makedirs(args.save_dir, exist_ok=True)
    test_list = os.listdir(args.render_dir)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    for name in test_list:
        render = cv2.imread(os.path.join(args.render_dir, name), -1)[:, :, :3]
        uv = cv2.resize(cv2.imread(os.path.join(args.uv_dir, name), -1)[:, :, :3], (args.input_size // 2, args.input_size // 2))
        render = cv2.copyMakeBorder(render, args.input_size // 2, args.input_size // 2, args.input_size // 2, args.input_size // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        render_img = Image.fromarray(render[:, :, ::-1])
        render_img = transform(render_img).unsqueeze(0).to(device)
        uv_img = Image.fromarray(uv[:, :, ::-1])
        uv_img = transform(uv_img).unsqueeze(0).to(device)
        with torch.no_grad():
            sample = generator(render_img, uv_img)
            outimg = np.clip(sample.cpu().numpy()[0].transpose((1, 2, 0)) * 127.5 + 127.5, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.save_dir, name), outimg[:, :, ::-1])
        print('Complete:', name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVP trainer")
    
    parser.add_argument("--render_dir", type=str, default=None, help="path to a test input")
    parser.add_argument("--uv_dir", type=str, default=None, help="path to a test input")
    parser.add_argument("--batch", type=int, default=1, help="batch sizes for each gpus")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--save_dir", type=str, default='test', help="path to the save folder")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--for_cpp", action="store_true", help="put BHWC to BCHW and scale into the model")
    args = parser.parse_args()

    device = "cuda"

    args.n_mlp = 4
    args.latent_frame = 64
    args.input_size = 256
    args.output_size = 1024
    
    g_ema = DoubleStyleUnet(args.input_size, args.output_size, args.latent_frame, args.n_mlp, channel_multiplier=args.channel_multiplier, for_cpp=args.for_cpp).to(device)
    g_ema.eval()

    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt)
    g_ema.load_state_dict(ckpt["g_ema"], strict=True)
    g_ema.assigen(ckpt)

    test(args, g_ema, device)

