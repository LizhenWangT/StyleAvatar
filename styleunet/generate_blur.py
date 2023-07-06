import numpy as np
import cv2
import random


jpg_num = 0

def generate_random_blur(image, poss=0):
    global jpg_num
    if poss == 0:
        poss = random.random()
    if poss < 0.05:
        image = cv2.blur(image, (10, 10))
    elif poss < 0.10:
        image = cv2.blur(image, (20, 20))
    elif poss < 0.20:
        image = cv2.blur(image, (40, 40))
    elif poss < 0.3:
        if jpg_num < 10:
            jpg_num += 1
        else:
            jpg_num = 0
        cv2.imwrite('logs/sample/tmp' + str(jpg_num) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
        image = cv2.imread('logs/sample/tmp' + str(jpg_num) + '.jpg', -1)
    elif poss < 0.4:
        if jpg_num < 10:
            jpg_num += 1
        else:
            jpg_num = 0
        cv2.imwrite('logs/sample/tmp' + str(jpg_num) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        image = cv2.imread('logs/sample/tmp' + str(jpg_num) + '.jpg', -1)
    elif poss < 0.5:
        if jpg_num < 10:
            jpg_num += 1
        else:
            jpg_num = 0
        cv2.imwrite('logs/sample/tmp' + str(jpg_num) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        image = cv2.imread('logs/sample/tmp' + str(jpg_num) + '.jpg', -1)
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


img = cv2.imread('E:/datasets/FFHQ/FFHQ/00001.png')
for i in range(50):
    mask = generate_random_blur(img, i / 50)
    cv2.imwrite('test_data/superresolution' + str(i) + '.png', mask)
