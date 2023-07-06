import numpy as np
import cv2


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def face_align(img, landmarks):
    # mediapipe landmarks
    border = 500
    square_lenght = 1024
    center = landmarks[197] + border
    #img[int(landmarks[197, 1]), int(landmarks[197, 0])] *= 0
    mouth_c = (landmarks[13] + landmarks[14]) / 2 + border
    length = distance(mouth_c, center) * 4.2 / 2
    img_border = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
    up_direction = (center - mouth_c) / distance(mouth_c, center)
    left_direction = np.array([-up_direction[1], up_direction[0]])
    p0 = center + up_direction * length + left_direction * length
    p1 = center + up_direction * length - left_direction * length
    p2 = center - up_direction * length - left_direction * length
    p3 = center - up_direction * length + left_direction * length
    src_pts = np.float32([[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]])
    dst_pts = np.float32([[0, 0], [square_lenght, 0], [square_lenght, square_lenght], [0, square_lenght]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img_border, M, (square_lenght, square_lenght))[:, ::-1]
    return result, (src_pts, dst_pts)


def face_align_inverse(img, align, param, kernel_size=15):
    border = 500
    img_height, img_width, _ = img.shape
    src_pts, dst_pts = param
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)
    result = cv2.warpPerspective(align[:, ::-1], M, (img_width + border * 2, img_height + border * 2))[500:-500, 500:-500]
    mask = (result > 0).astype(np.uint8)
    # avoid black edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.erode(mask, kernel, iterations=2)
    img = img * (1 - mask) + result * mask
    return img


if __name__ == '__main__':
    img = cv2.imread('test_data/1.png')
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(img)
    if not results.multi_face_landmarks:
        print('No Face Detected!')
    landmarks = results.multi_face_landmarks[0]
    landmarks = np.array([(lmk.x, lmk.y) for lmk in landmarks.landmark])
    landmarks[:, 0] *= img.shape[1]
    landmarks[:, 1] *= img.shape[0]
    align, param = face_align(img, landmarks)
    cv2.imwrite('results/1a.png', align[:, :, ::-1])
    img = face_align_inverse(img, align, param)
    cv2.imwrite('results/1b.png', img[:, :, ::-1])

