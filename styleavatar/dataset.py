import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import random
from torch_utils import get_embedder


class TrainDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.img_list = os.listdir(os.path.join(self.path, 'render'))
        self.img_list.sort()
        self.length = len(self.img_list)

        self.transform = transform
        self.base = self.path.split(os.sep)[-1]
        self.video_index = 0
        self.embedder = get_embedder(32)
        print('load dataset:', self.video_index, self.base)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, 'image', self.img_list[index]))
        image = self.transform(image)
        uv = Image.open(os.path.join(self.path, 'uv', self.img_list[index]))
        uv = self.transform(uv)
        render = Image.open(os.path.join(self.path, 'render', self.img_list[index]))
        render = self.transform(render)
        back = Image.open(os.path.join(self.path, 'back', self.img_list[index])).convert('RGB')
        back = self.transform(back)
        frame_latent = self.embedder(torch.tensor([index / 5000], dtype=torch.float32))
        video_latent = self.embedder(torch.tensor([self.video_index / 10], dtype=torch.float32))
        return image, uv, render, back, frame_latent, video_latent


class PreTrainDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.namelist = ['0', '1', '2', '3', '4', '5']
        self.img_list = []
        self.nums_list = []
        tmp_num = 0
        for name in self.namelist:
            self.img_list += os.listdir(os.path.join(self.path, name, 'render'))
            tmp_num += len(os.listdir(os.path.join(self.path, name, 'render')))
            self.nums_list.append(tmp_num)
        self.length = len(self.img_list)

        self.transform = transform
        self.base = self.path.split(os.sep)[-1]
        self.embedder = get_embedder(32)
        print('load dataset:', self.base, self.nums_list, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_index = 0
        for pnum in self.nums_list:
            if pnum < index:
                video_index += 1
        name = self.namelist[video_index]
        image = Image.open(os.path.join(self.path, name, 'image', self.img_list[index]))
        image = self.transform(image)
        uv = Image.open(os.path.join(self.path, name, 'uv', self.img_list[index]))
        uv = self.transform(uv)
        render = Image.open(os.path.join(self.path, name, 'render', self.img_list[index]))
        render = self.transform(render)
        back = Image.open(os.path.join(self.path, name, 'back', self.img_list[index])).convert('RGB')
        back = self.transform(back)
        frame_latent = self.embedder(torch.tensor([(index - self.nums_list[video_index]) / 5000], dtype=torch.float32))
        video_latent = self.embedder(torch.tensor([video_index / 10 - 0.5], dtype=torch.float32))
        return image, uv, render, back, frame_latent, video_latent
