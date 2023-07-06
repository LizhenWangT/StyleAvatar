# for single GPU
# train, mode 0 for face inpainting; 1 for face superresolution; 2 for face retouching; 3 for 3dmm-to-portriat image
python train.py --batch 3 --mode 2 --augment --augment_p 0.001
python train.py --batch 3 --ckpt pretrained/xxxx.pt --mode 0 --augment --augment_p 0.01 path-to-dataset
python train.py --batch 3 --ckpt pretrained/xxxx.pt --mode 1 --augment --augment_p 0.01 path-to-dataset
python train.py --batch 3 --ckpt pretrained/xxxx.pt --mode 2 --augment --augment_p 0.01 path-to-dataset
python train.py --batch 3 --ckpt pretrained/xxxx.pt --mode 3 path-to-dataset

# test, --skin_whiten 0-1 if you want, --use_alignment if your input image is not a 1024x1024 aligned portriat image, --iter 1~3 for iterations of face retouching (only for mode 2)
python test.py --input_dir path-to-image-folder --ckpt logs/checkpoint/xxxx.pt --save_dir path-to-save-folder --skin_whiten 0.0 --mode 0 --use_alignment

# train, for multi-GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port='1234' train.py --batch 3 --mode 0 --ckpt pretrained/xxxx.pt --augment --augment_p 0.2 /media/data1/wlz/datasets/

# torch2onnx, --for_cpp will add BGR2RGB, BHWC2BCHW, (0,255)-to-(-1,1) into the model 
python torch2onnx.py --test_img path-to-dataset/render/000000.png --ckpt logs/checkpoint/tdmm_xxxxxx.pt --save_name xxx --mode 3 --for_cpp

# onnx2trt on Windows
tensorrt/trtexec.exe --onnx=pretrained/xxx.onnx --saveEngine=pretrained/xxx_16.engine --fp16
