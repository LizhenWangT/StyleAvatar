import os
import numpy as np
import argparse
import time
import cv2

import torch
from torch.nn import functional as F

from networks.generator_for_onnx import StyleUNet
from torchvision import transforms, utils
from PIL import Image


def torch2onnx(args, generator, device):
    with torch.no_grad():
        if args.test_img is not None:
            if args.for_cpp:
                test_img = cv2.imread(args.test_img, -1)
                test_img = torch.from_numpy(test_img).type(torch.float32).unsqueeze(0).repeat(args.batch, 1, 1, 1).to(device)
            else:
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
                test_img = Image.open(args.test_img)
                test_img = transform(test_img).unsqueeze(0).repeat(args.batch, 1, 1, 1).to(device)
            latent = torch.randn(args.batch, args.style_dim).to(device)
            sample = generator(test_img, latent)
            if args.for_cpp:
                sample = sample.cpu().numpy().astype(np.uint8)
                cv2.imwrite("test_pytorch.png", sample[0])
            else:
                sample = sample.clamp(min=-1, max=1)
                utils.save_image(sample, f"test_pytorch.png", nrow=1, normalize=True, range=(-1, 1))
            # model run time test
            # warm up
            with torch.no_grad():
                for _ in range(10):
                    sample = generator(test_img, latent)
                # start
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                for _ in range(100):
                    sample = generator(test_img, latent)
                end.record()
                torch.cuda.synchronize()
            print('run time:', start.elapsed_time(end) / 100, 'ms')
        else:
            if args.for_cpp:
                test_img = torch.zeros(args.batch, args.input_size, args.input_size, 3).to(device)
            else:
                test_img = torch.zeros(args.batch, 3, args.input_size, args.input_size).to(device)
            latent = torch.randn(args.batch, args.style_dim).to(device)

        export_onnx_file = args.save_name + ".onnx"
        torch.onnx.export(generator, (test_img, latent), export_onnx_file, verbose=True, opset_version=12, keep_initializers_as_inputs=True,
                        do_constant_folding=True, export_params=True, input_names=["cond_img", "latent"], output_names=["output"], dynamic_axes=None)
        print('Complete!')

    if args.test_img is not None:
        import onnx
        import onnxruntime as ort
        model = onnx.load(export_onnx_file)
        # Check that the model is well formed
        onnx.checker.check_model(model)
        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))
        ort_session = ort.InferenceSession(export_onnx_file)
        input_name = ort_session.get_inputs()
        print('inputs:', input_name)
        start = time.time()
        for _ in range(10):
            outputs = ort_session.run(["output"], {"cond_img":test_img.cpu().numpy(), "latent":latent.cpu().numpy()})
        print('run time:', (time.time() - start) / 10 * 1000, 'ms')
        if args.for_cpp:
            cv2.imwrite("test_onnx.png", outputs[0][0].astype(np.uint8))
        else:
            imgsave = (np.clip(outputs[0][0].transpose((1, 2, 0)) * 127.5 + 127.5, 0, 255)).astype(np.uint8)
            cv2.imwrite('test_onnx.png', imgsave[:, :, ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVP trainer")
    
    parser.add_argument("--test_img", type=str, default=None, help="path to a test input")
    parser.add_argument("--batch", type=int, default=1, help="batch sizes for each gpus")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--save_name", type=str, default='onnxmodel', help="path to the checkpoints to resume training")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--mode", type=int, default=0, help="0 for inpainting; 1 for superresolution; 2 for retouching; 3 for 3dmm")
    parser.add_argument("--for_cpp", action="store_true", help="put BHWC to BCHW and scale into the model")
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
    
    g_ema = StyleUNet(args.input_size, args.output_size, args.style_dim, args.n_mlp, args.channel_multiplier, device=device, for_cpp=args.for_cpp).to(device)
    g_ema.eval()

    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt)
    g_ema.load_state_dict(ckpt["g_ema"], strict=True)

    torch2onnx(args, g_ema, device)

