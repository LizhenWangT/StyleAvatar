import numpy as np
import cv2
from PIL import Image, ImageEnhance

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


def facial_dermabrasion_effect(img, k=1):
    imgout = cv2.bilateralFilter(img, 31, 75, 75)
    imgout = cv2.addWeighted(img, 0.3, imgout, 0.7, 0)
    imgout = Image.fromarray(imgout)
    imgout = ImageEnhance.Sharpness(imgout)
    imgout = imgout.enhance(1.5)
    imgout = ImageEnhance.Contrast(imgout)
    imgout = imgout.enhance(1.15)
    imgout = np.array(imgout)
    imgout = img.astype(np.float32) * (1 - k) + imgout.astype(np.float32) * k
    return imgout.astype(np.uint8)


img = cv2.imread('E:/datasets/FFHQ/FFHQ/00001.png')
cv2.imwrite('0.png', img)
mask = skin_whiten(img, 0.5)
cv2.imwrite('1.png', mask)
mask = skin_whiten(img, 1, 2)
cv2.imwrite('2.png', mask)
mask = facial_dermabrasion_effect(img, 1)
cv2.imwrite('3.png', mask)
