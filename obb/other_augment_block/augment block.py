import numpy as np
import cv2
import os

def improve_Retinex(img, sigma_list,w1,w2,w3,radius):
    img = np.float64(img) + 1.0
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    img_retinex = retinex / len(sigma_list)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.float64(img_retinex)
    G1 = cv2.GaussianBlur(img_retinex, (radius, radius), 1.0)

    G2 = cv2.GaussianBlur(img_retinex, (radius * 2 - 1, radius * 2 - 1), 2.0)

    G3 = cv2.GaussianBlur(img_retinex, (radius * 4 - 1, radius * 4 - 1), 4.0)


    D1 = (1 - w1 * np.sign(img_retinex - G1)) * (img_retinex - G1)

    D2 = w2 * (G1 - G2)

    D3 = w3 * (G2 - G3)

    img_mss = img_retinex + D1 + D2 + D3
    img_retinex = cv2.convertScaleAbs(img_mss)
    img_retinex = np.uint8(img_retinex)

    return img_retinex





if __name__ == '__main__':
    data_path = r'data'
    img_list = os.listdir(data_path)
    config = {
    "sigma_list": [15, 80, 250],
    "w1": 0.5,
    "w2": 0.5,
    "w3": 0.25,
    "radius": 3
    }
    if len(img_list) == 0:
        print
        'Data directory is empty.'
        exit()


    for img_name in img_list:
        if img_name == '.gitkeep':
            continue

        img = cv2.imread(os.path.join(data_path, img_name))

        img_enhence = improve_Retinex(
            img,
            config['sigma_list'],
            config['w1'],
            config['w2'],
            config['w3'],
            config['radius'],
        )

        cv2.imshow('Image', img)
        cv2.imshow('img_enhence', img_enhence)
        cv2.waitKey()

