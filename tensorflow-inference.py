import keras
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog

model = keras.models.load_model('model.keras')


#hog params
orientation_bins = 16
pixel_window = 32
input_len = int(128 * 64 * orientation_bins / (pixel_window ** 2))


# Moving window part :D
test_image = cv2.imread("people_2.jpg")
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
h_test, v_test = np.shape(test_image)
sv = 64         # image dims
sh = 128
step = 8        # window slide

scale = 1
scale_step = 0.8
levels = 5

output_level_maps = []

for level in range(1, levels):
    print("starting level: ", level, "\t scale: ", scale)
    h, v = [int(h_test * scale), int(v_test * scale)]
    curr_img = cv2.resize(test_image, (v, h))

    count_h = int((h - sh) / step)
    count_v = int((v - sv) / step)

    out_map = np.zeros((count_h, count_v), dtype=float)
    for i in range(count_h):
        for j in range(count_v):
            horizontal = int(i * step)
            vertical = int(j * step)

            sub_img = curr_img[horizontal:horizontal+sh, vertical:vertical+sv]
            ft = hog(sub_img, orientations=orientation_bins, pixels_per_cell=(pixel_window, pixel_window), cells_per_block=(1, 1))
            ft = np.reshape(ft, (1, len(ft)))

            out_map[i, j] = model.predict(ft, verbose=0)[0]

    out_map[out_map < 0.5] = 0

    scale = scale * scale_step
    output_level_maps.append(out_map)

