import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
import os
import numpy as np
import random


class InpaitingDataset(Dataset):
    def __init__(self, path, transform):
        self.img_list = os.listdir(os.path.join(path, 'FFHQ'))
        self.img_list.sort()
        self.length = len(self.img_list)

        self.path = path
        self.transform = transform
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

    def __len__(self):
        return self.length

    def generate_face_keypoints_convex_hull_mask(self, image, select=False):
        results = self.mp_face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([(lmk.x, lmk.y) for lmk in landmarks.landmark])
        landmarks[:, 0] *= image.shape[1]
        landmarks[:, 1] *= image.shape[0]
        if select:
            landmarks = landmarks[np.random.randint(0, 467, size=10)]
        convex_hull = cv2.convexHull(landmarks.astype(np.int32))
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [convex_hull], 0, 255, -1)
        return mask

    def generate_random_mask(self, image, shape, max_sides=10, max_size=50):
        mask = np.zeros(shape, dtype=np.uint8)
        poss = random.random()
        if poss < 0.2:
            num_shapes = np.random.randint(max_sides) + 3
            vertices = np.random.randint(max_size, size=(num_shapes, 2))
            cv2.fillPoly(mask, [vertices], 255)
        elif poss < 0.4:
            center = (np.random.randint(shape[1]), np.random.randint(shape[0]))
            radius = np.random.randint(10, max_size//3)
            cv2.circle(mask, center, radius, (255, 255, 255), -1)
        elif poss < 0.6:
            mask = self.generate_face_keypoints_convex_hull_mask(image, select=True)
            if mask is None:
                return image
        elif poss < 0.8:
            mask = self.generate_face_keypoints_convex_hull_mask(image)
            if mask is None:
                return image
            mask = 255 - mask
            if random.random() > 0.9:
                return image * (mask > 0).astype(np.uint8)[:, :, None]
        else:
            return image
        return image * (1 - (mask > 0).astype(np.uint8))[:, :, None]

    def __getitem__(self, index):
        origin = cv2.imread(os.path.join(self.path, 'FFHQ', self.img_list[index]), -1)[:, :, :3]
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        origin_image = Image.fromarray(origin)
        origin_image = self.transform(origin_image)
        mask_image = self.generate_random_mask(origin, [1024, 1024], 10, 1024)
        mask_image = Image.fromarray(mask_image)
        mask_image = self.transform(mask_image)

        return origin_image, mask_image


class SuperResolutionDataset(Dataset):
    def __init__(self, path, transform):
        self.img_list = os.listdir(os.path.join(path, 'FFHQ'))
        self.img_list.sort()
        self.length = len(self.img_list)

        self.path = path
        self.transform = transform
        self.jpg_num = 0

    def __len__(self):
        return self.length

    def generate_random_blur(self, image):
        poss = random.random()
        if poss < 0.05:
            image = cv2.blur(image, (10, 10))
        elif poss < 0.10:
            image = cv2.blur(image, (20, 20))
        elif poss < 0.20:
            image = cv2.blur(image, (40, 40))
        elif poss < 0.3:
            if self.jpg_num < 10:
                self.jpg_num += 1
            else:
                self.jpg_num = 0
            cv2.imwrite('logs/sample/tmp' + str(self.jpg_num) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
            image = cv2.imread('logs/sample/tmp' + str(self.jpg_num) + '.jpg', -1)
        elif poss < 0.4:
            if self.jpg_num < 10:
                self.jpg_num += 1
            else:
                self.jpg_num = 0
            cv2.imwrite('logs/sample/tmp' + str(self.jpg_num) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
            image = cv2.imread('logs/sample/tmp' + str(self.jpg_num) + '.jpg', -1)
        elif poss < 0.5:
            if self.jpg_num < 10:
                self.jpg_num += 1
            else:
                self.jpg_num = 0
            cv2.imwrite('logs/sample/tmp' + str(self.jpg_num) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
            image = cv2.imread('logs/sample/tmp' + str(self.jpg_num) + '.jpg', -1)
        elif poss < 0.6:
            gaussian_noise = np.random.randn(*image.shape) * 20
            sparse_matrix = np.random.rand(*image.shape)
            sparse_matrix[sparse_matrix < 0.97] = 0
            image = cv2.add(image, (gaussian_noise * sparse_matrix).astype(np.uint8))
        elif poss < 0.65:
            downsampled_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            image = cv2.resize(downsampled_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        elif poss < 0.7:
            downsampled_img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
            image = cv2.resize(downsampled_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        elif poss < 0.75:
            downsampled_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            image = cv2.resize(downsampled_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        elif poss < 0.8:
            downsampled_img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
            image = cv2.resize(downsampled_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        elif poss < 0.85:
            downsampled_img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
            image = cv2.resize(downsampled_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return image

    def __getitem__(self, index):
        origin = cv2.imread(os.path.join(self.path, 'FFHQ', self.img_list[index]), -1)[:, :, :3]
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        origin_image = Image.fromarray(origin)
        origin_image = self.transform(origin_image)
        blur_image = self.generate_random_blur(origin)
        blur_image = Image.fromarray(blur_image)
        blur_image = self.transform(blur_image)

        return origin_image, blur_image


class RetouchDataset(Dataset):
    def __init__(self, path, transform):
        self.img_list = os.listdir(os.path.join(path, 'FFHQ_retouch'))
        self.img_list.sort()
        self.length = len(self.img_list)

        self.path = path
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        retouch = Image.open(os.path.join(self.path, 'FFHQ_retouch', self.img_list[index])).convert('RGB')
        origin = Image.open(os.path.join(self.path, 'FFHQ', self.img_list[index]))
        retouch = self.transform(retouch)
        origin = self.transform(origin)

        return retouch, origin


class TDMMDataset(Dataset):
    def __init__(self, path, transform):
        self.img_list = os.listdir(os.path.join(path, 'render'))
        self.img_list.sort()
        self.length = len(self.img_list)

        self.path = path
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        tdmm = Image.open(os.path.join(self.path, 'render', self.img_list[index])).convert('RGB')
        out = Image.open(os.path.join(self.path, 'image', self.img_list[index])).convert('RGB')
        tdmm = self.transform(tdmm)
        out = self.transform(out)

        return out, tdmm
