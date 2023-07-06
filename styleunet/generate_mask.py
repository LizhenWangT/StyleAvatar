import numpy as np
import cv2
import random
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

def generate_face_keypoints_convex_hull_mask(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (640, 480))
    results = mp_face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    landmarks = np.array([(lmk.x, lmk.y) for lmk in landmarks.landmark])
    landmarks[:, 0] *= image.shape[1]
    landmarks[:, 1] *= image.shape[0]
    convex_hull = cv2.convexHull(landmarks.astype(np.int32))
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [convex_hull], 0, 255, -1)
    return mask


def generate_random_mask(image, shape, max_sides=10, max_size=50):
    mask = np.zeros(shape, dtype=np.uint8)
    poss = random.random()
    if poss < 0.3:
        num_shapes = np.random.randint(max_sides) + 3
        vertices = np.random.randint(max_size, size=(num_shapes, 2))
        cv2.fillPoly(mask, [vertices], 255)
    elif poss < 0.6:
        center = (np.random.randint(shape[1]), np.random.randint(shape[0]))
        radius = np.random.randint(10, max_size//3)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
    elif poss < 0.8:
        mask = 255 - generate_face_keypoints_convex_hull_mask(image)
    if random.random() > 0.9:
        image = image * (mask > 0).astype(np.uint8)[:, :, None]
    else:
        image = image * (1 - (mask > 0).astype(np.uint8))[:, :, None]
    return image


img = cv2.imread('test_data/retouch/wlz.png')#('E:/datasets/FFHQ/FFHQ/00001.png')
for i in range(50):
    mask = generate_random_mask(img, [1024, 1024], 10, 1024)
    cv2.imwrite('test_data/inpainting/wlz' + str(i) + '.png', mask)
