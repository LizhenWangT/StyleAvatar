# StyleAvatar

[Lizhen Wang](https://lizhenwangt.github.io/), Xiaochen Zhao, [Jingxiang Sun](https://mrtornado24.github.io/), Yuxiang Zhang, [Hongwen Zhang](https://hongwenzhang.github.io/), [Tao Yu](https://ytrock.com/), [Yebin Liu](http://www.liuyebin.com/)

ACM SIGGRAPH 2023 Conference Proceedings

Tsinghua University & NNKOSMOS

[[Arxiv]](https://arxiv.org/abs/2305.00942) [[Paper]](https://www.liuyebin.com/styleavatar/assets/StyleAvatar.pdf) [[Project Page]](https://www.liuyebin.com/styleavatar/styleavatar.html) [[Demo Video]](https://www.liuyebin.com/styleavatar/assets/Styleavatar.mp4)

<div align=center><img src="./docs/teaser.jpg"></div>

### Abstract

>Face reenactment methods attempt to restore and re-animate portrait videos as realistically as possible. Existing methods face a dilemma in quality versus controllability: 2D GAN-based methods achieve higher image quality but suffer in fine-grained control of facial attributes compared with 3D counterparts. In this work, we propose StyleAvatar, a real-time photo-realistic portrait avatar reconstruction method using StyleGAN-based networks, which can generate high-fidelity portrait avatars with faithful expression control. We expand the capabilities of StyleGAN by introducing a compositional representation and a sliding window augmentation method, which enable faster convergence and improve translation generalization. Specifically, we divide the portrait scenes into three parts for adaptive adjustments: facial region, non-facial foreground region, and the background. Besides, our network leverages the best of UNet, StyleGAN and time coding for video learning, which enables high-quality video generation. Furthermore, a sliding window augmentation method together with a pre-training strategy are proposed to improve translation generalization and training performance, respectively. The proposed network can converge within two hours while ensuring high image quality and a forward rendering time of only 20 milliseconds. Furthermore, we propose a real-time live system, which further pushes research into applications. Results and experiments demonstrate the superiority of our method in terms of image quality, full portrait video generation, and real-time re-animation compared to existing facial reenactment methods.

<div align=center><img src="./docs/results.jpg" width = 70%></div>
<center>**Fig.1** Facial re-enactment results of StyleAvatar.</center>

<div align=center><img src="./docs/pipeline.jpg"></div>
<center>**Fig.2** The pipeline of our method.</center>

## Change Log

2023.05.05 We will release code and pre-trained model in May.

## Requirements
- Python 3.8
- PyTorch 1.10.0
- torchvision 0.11.0
- Cuda 11.3
- OpenCV
- Numpy
- tqdm
- ninja

You need to compile the ops provided by [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) using ninja:

```
cd third_libs/stylegan_ops
python3 setup.py install
```

## Pretrain Model
coming soon

## Train your own model
coming soon

## Citation

If you use our code for your research, please consider citing:
```
@inproceedings{wang2023styleavatar,
  title={StyleAvatar: Real-time Photo-realistic Portrait Avatar from a Single Video},
  author={Wang, Lizhen and Zhao, Xiaochen and Sun, Jingxiang and Zhang, Yuxiang and Zhang, Hongwen and Yu, Tao and Liu, Yebin},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  pages={},
  year={2023}
}
```

## Acknowledgement & License
The code is partially borrowed from [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch). And many thanks to the volunteers participated in data collection. Our License can be found in [LICENSE](./LICENSE).

