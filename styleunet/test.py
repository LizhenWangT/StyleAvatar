import os
import numpy as np
import argparse
import cv2

import torch
from torch.nn import functional as F

from networks.generator_for_onnx import StyleUNet
from torchvision import transforms, utils
from PIL import Image


def skin_whiten(img, k=0.9, iter=1):
    imgout = img.copy()
    cl = np.array([
	1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39,
	41, 43, 44, 46, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 73, 74,
	76, 78, 79, 81, 83, 84, 86, 87, 89, 91, 92, 94, 95, 97, 99, 100, 102, 103, 105,
	106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 128,
	130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 147, 149, 150,
	151, 153, 154, 155, 156, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170,
	171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
	188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
	204, 205, 205, 206, 207, 208, 209, 210, 211, 211, 212, 213, 214, 215, 215, 216,
	217, 218, 219, 219, 220, 221, 222, 222, 223, 224, 224, 225, 226, 226, 227, 228,
	228, 229, 230, 230, 231, 232, 232, 233, 233, 234, 235, 235, 236, 236, 237, 237,
	238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 244, 245,
	245, 246, 246, 246, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 250,
	251, 251, 251, 251, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254,
	254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 256]) - 1
    for _ in range(iter):
        imgout = cl[imgout]
    imgout = img.astype(np.float32) * (1 - k) + imgout.astype(np.float32) * k
    return imgout.astype(np.uint8)


def test(args, generator, device):
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_alignment:
        import mediapipe as mp
        from face_alignment import face_align, face_align_inverse
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
    latent = torch.randn(args.batch, args.style_dim).to(device)
    test_list = os.listdir(args.input_dir)
    for name in test_list:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
        test_img = cv2.imread(os.path.join(args.input_dir, name), -1)[:, :, :3]
        if args.skin_whiten > 0.0:
            test_img = skin_whiten(test_img, args.skin_whiten)
        if args.use_alignment:
            results = mp_face_mesh.process(test_img[:, :, ::-1])
            if not results.multi_face_landmarks:
                print('No Face Detected in', name)
                continue
            landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([(lmk.x, lmk.y) for lmk in landmarks.landmark])
            landmarks[:, 0] *= test_img.shape[1]
            landmarks[:, 1] *= test_img.shape[0]
            align, param = face_align(test_img[:, :, ::-1], landmarks)
        else:
            if args.mode != 3 and (test_img.shape[0] != 1024 or test_img.shape[1] != 1024):
                test_img = cv2.resize(test_img, (1024, 1024))
            align = test_img[:, :, ::-1]
        input_img = Image.fromarray(align)
        input_img = transform(input_img).unsqueeze(0).repeat(args.batch, 1, 1, 1).to(device)
        with torch.no_grad():
            sample = generator(input_img, latent)
            for _ in range(args.iter - 1):
                sample = generator(sample, latent)
            #utils.save_image(sample, os.path.join(args.save_dir, name), nrow=1, normalize=True, range=(-1, 1))
            outimg = np.clip(sample.cpu().numpy()[0].transpose((1, 2, 0)) * 127.5 + 127.5, 0, 255).astype(np.uint8)
        if args.use_alignment:
            outimg = face_align_inverse(test_img[:, :, ::-1], outimg, param)
        #if args.skin_whiten > 0.0:
        #    outimg = skin_whiten(outimg, args.skin_whiten)
        cv2.imwrite(os.path.join(args.save_dir, name), outimg[:, :, ::-1])
        print('Complete:', name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVP trainer")
    
    parser.add_argument("--input_dir", type=str, default=None, help="path to a test input")
    parser.add_argument("--batch", type=int, default=1, help="batch sizes for each gpus")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--save_dir", type=str, default='test', help="path to the save folder")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--skin_whiten", type=float, default=0.0, help="strength of face whiten")
    parser.add_argument("--use_alignment", action="store_true", help="use face alignment")
    parser.add_argument("--iter", type=int, default=1, help="network forward iterations")
    parser.add_argument("--mode", type=int, default=0, help="0 for inpainting; 1 for superresolution; 2 for retouching; 3 for 3dmm")
    args = parser.parse_args()

    device = "cuda"

    args.n_mlp = 4
    args.style_dim = 64
    if args.mode == 0:
        args.input_size = 1024
        args.output_size = 1024
    elif args.mode == 1:
        args.input_size = 1024
        args.output_size = 1024
    elif args.mode == 2:
        args.input_size = 1024
        args.output_size = 1024
    elif args.mode == 3:
        args.input_size = 256
        args.output_size = 1024
    
    g_ema = StyleUNet(args.input_size, args.output_size, args.style_dim, args.n_mlp, args.channel_multiplier, device=device).to(device)
    g_ema.eval()

    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt)
    g_ema.load_state_dict(ckpt["g_ema"], strict=True)

    test(args, g_ema, device)

