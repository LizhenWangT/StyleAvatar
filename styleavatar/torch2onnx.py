import os
import numpy as np
import argparse
import time
import cv2

import torch
from torch.nn import functional as F

from torchvision import transforms, utils
from PIL import Image


def torch2onnx(args, generator, device):
    with torch.no_grad():
        if args.test_img is not None:
            if args.for_cpp:
                test_img = cv2.imread(args.test_img, -1)
                test_img = cv2.copyMakeBorder(test_img, args.input_size // 2, args.input_size // 2, args.input_size // 2, args.input_size // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                test_img = torch.from_numpy(test_img).type(torch.float32).unsqueeze(0).repeat(args.batch, 1, 1, 1).to(device)
                uv_img = cv2.imread(args.uv_img, -1)
                uv_img = cv2.resize(uv_img, (args.input_size // 2, args.input_size // 2))
                uv_img = torch.from_numpy(uv_img).type(torch.float32).unsqueeze(0).repeat(args.batch, 1, 1, 1).to(device)
            else:
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
                test_img = cv2.imread(args.test_img, -1)[:, :, ::-1]
                test_img = cv2.copyMakeBorder(test_img, args.input_size // 2, args.input_size // 2, args.input_size // 2, args.input_size // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                test_img = Image.fromarray(test_img)#open(args.test_img)
                test_img = transform(test_img).unsqueeze(0).repeat(args.batch, 1, 1, 1).to(device)
                uv_img = Image.open(args.uv_img)
                uv_img = uv_img.resize((args.input_size // 2, args.input_size // 2))
                uv_img = transform(uv_img).unsqueeze(0).repeat(args.batch, 1, 1, 1).to(device)
            sample = generator(test_img, uv_img)
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
                    sample = generator(test_img, uv_img)
                # start
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                for _ in range(100):
                    sample = generator(test_img, uv_img)
                end.record()
                torch.cuda.synchronize()
            print('run time:', start.elapsed_time(end) / 100, 'ms')
        else:
            if args.for_cpp:
                test_img = torch.zeros(args.batch, args.input_size, args.input_size, 3).to(device)
                uv_img = torch.zeros(args.batch, args.input_size // 2, args.input_size // 2, 3).to(device)
            else:
                test_img = torch.zeros(args.batch, 3, args.input_size, args.input_size).to(device)
                uv_img = torch.zeros(args.batch, args.input_size // 2, args.input_size // 2, 3).to(device)

        export_onnx_file = args.save_name + ".onnx"
        torch.onnx.export(generator, (test_img, uv_img), export_onnx_file, verbose=True, opset_version=16, keep_initializers_as_inputs=True,
                        do_constant_folding=True, export_params=True, input_names=["image", "uv"], output_names=["output"], dynamic_axes=None)
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
            outputs = ort_session.run(["output"], {"image":test_img.cpu().numpy(), "uv":uv_img.cpu().numpy()})
        print('run time:', (time.time() - start) / 10 * 1000, 'ms')
        if args.for_cpp:
            cv2.imwrite("test_onnx.png", outputs[0][0].astype(np.uint8))
        else:
            imgsave = (np.clip(outputs[0][0].transpose((1, 2, 0)) * 127.5 + 127.5, 0, 255)).astype(np.uint8)
            cv2.imwrite('test_onnx.png', imgsave[:, :, ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVP trainer")
    
    parser.add_argument("--test_img", type=str, default=None, help="path to a test input")
    parser.add_argument("--uv_img", type=str, default=None, help="path to a test input")
    parser.add_argument("--batch", type=int, default=1, help="batch sizes for each gpus")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--save_name", type=str, default='onnxmodel', help="path to the checkpoints to resume training")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--for_cpp", action="store_true", help="put BHWC to BCHW and scale into the model")
    args = parser.parse_args()

    device = "cuda"

    args.n_mlp = 4
    args.latent_frame = 64
    args.latent_video = 64
    args.input_size = 256
    args.feature_size = args.input_size // 2
    args.output_size = 1024
    
    from networks.generator_for_onnx import DoubleStyleUnet
    g_ema = DoubleStyleUnet(args.input_size, args.output_size, args.latent_frame, args.n_mlp, channel_multiplier=args.channel_multiplier, for_cpp=args.for_cpp).to(device)
    g_ema.eval()

    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt)
    g_ema.load_state_dict(ckpt["g_ema"], strict=True)
    g_ema.assigen(ckpt)

    torch2onnx(args, g_ema, device)

