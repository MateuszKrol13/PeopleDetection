# Project implementation in sklearn and tensorflow for comparisson
# Windows fix for matplotlib
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

# my functions
from functions import *
from HOG import get_HOG_features

# image processing libs
import numpy as np
import keras
from sklearn.utils import shuffle
from skimage.feature import hog
import cv2

# QOL libs
import glob
import tqdm

# epmpty data
x = []
y = []

#hog params
orientation_bins = 16
pixel_window = 16

# Load data - list of numpy arrays
print("Getting people dataset features...")
people_x = []
people_y = []
for img in tqdm.tqdm(glob.glob("classes\\person\\*.png")):
    im = cv2.imread(img)
    im = cv2.resize(im, (64, 128))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #fd = hog(im, orientations=orientation_bins, pixels_per_cell=(pixel_window, pixel_window), cells_per_block=(2, 2))
    fd = get_HOG_features(im, pixels_per_cell=pixel_window, bin_count=orientation_bins, cells_per_block=2)
    people_x.append(fd)
    people_y.append(1)

people_x = people_x[:1000]
people_y = people_y[:1000]

print("Getting background dataset features...")
background_x = []
background_y = []
for img in tqdm.tqdm(glob.glob("classes\\background\\*.png")):
    im = cv2.imread(img)
    im = cv2.resize(im, (64, 128))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fd = get_HOG_features(im, pixels_per_cell=pixel_window, bin_count=orientation_bins, cells_per_block=2)
    background_x.append(fd)
    background_y.append(0)

x = people_x + background_x
y = people_y + background_y

# Shuffle using numpy -> think of etter way later on.
x_train, y_train = shuffle(np.asarray(x), np.asarray(y))

print("Starting Keras Model...")
# model
input_len = len(fd)
model = keras.Sequential(
    [
        keras.Input(shape=input_len),
        keras.layers.Dense(64, activation= "relu"),
        keras.layers.Dense(64, activation= "relu"),
        keras.layers.Dense(32, activation= "relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# save best callback
checkpoint_path = './models'
SaveBest = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_accuracy", mode="max",
                                           save_weights_only=True, save_best_only=True)
TensorBoard = keras.callbacks.TensorBoard(log_dir='logs/')

model.summary()
optimizer = keras.optimizers.SGD(learning_rate=0.1)
loss = keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=200, validation_split=0.3, callbacks=[SaveBest])
model.load_weights(checkpoint_path)

# Moving window part :D
test_image = cv2.imread("people_1.jpg")
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image_cpy = np.copy(test_image)
h_test, v_test = np.shape(test_image)
sv = 64         # image dims
sh = 128
step = 8        # window slide

scale = 1
scale_step = 0.8
levels = 5

output_level_maps = []
dets = []
scores=[]

for level in range(1, levels):
    print("starting level: ", level, "\t scale: ", scale)
    h, v = [int(h_test * scale), int(v_test * scale)]
    curr_img = cv2.resize(test_image, (v, h))

    count_h = int((h - sh) / step)
    count_v = int((v - sv) / step)

    ft_container = []
    for i in range(count_h):
        for j in range(count_v):
            horizontal = int(i * step)
            vertical = int(j * step)

            sub_img = curr_img[horizontal:horizontal+sh, vertical:vertical+sv]
            ft = get_HOG_features(sub_img, pixels_per_cell=pixel_window, bin_count=orientation_bins, cells_per_block=2)
            ft_container.append(ft)

    features_map = np.asarray(ft_container)
    out_map = model.predict(features_map).flatten()
    out_map = out_map.reshape((count_h, count_v))

    out_map[out_map < 0.5] = 0
    rs, cs = np.where(out_map > 0)

    for i in range(len(rs)):
        x = (cs[i] * step) / scale
        y = (rs[i] * step) / scale
        v = sv / scale
        h = sh / scale

        dets.append([x, y, v, h])
        scores.append(out_map[rs[i], cs[i]])

    scale = scale * scale_step
    output_level_maps.append(out_map)

for i, (x, y, w, h) in enumerate(dets):
    cv2.rectangle(test_image_cpy, (int(x), int(y)), (int(x + w), int(y + h)), 255, 2)

plt.figure(figsize=(10, 10))
plt.imshow(test_image_cpy, cmap='gray')
plt.title('Detections')
plt.show(block=True)


test_image = cv2.imread("people_1.jpg")
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# reduce boxes
filtered_dets =[]
filtered_scores = []
tmp_dets = np.asarray(dets)
tmp_scores = np.asarray(scores)

while True:
    if (0 == len(tmp_dets)):
        break
    i = np.argmax(tmp_scores)

    analysed_det = tmp_dets[i, :]
    analysed_score = tmp_scores[i]

    filtered_dets.append(analysed_det)
    filtered_scores.append(analysed_score)
    tmp_dets = np.delete(tmp_dets, i, 0)
    tmp_scores = np.delete(tmp_scores, i)

    iou_tab = []
    for det in tmp_dets:
        IoU = get_iou(analysed_det, det)
        iou_tab.append(IoU)

    # remove unwanted rois
    iou_tab = np.asarray(iou_tab)
    idx_to_delete = np.where(iou_tab > 0.01)
    tmp_dets = np.delete(tmp_dets, idx_to_delete, 0)
    tmp_scores = np.delete(tmp_scores, idx_to_delete)

for i, (x, y, w, h) in enumerate(filtered_dets):
    cv2.rectangle(test_image, (int(x), int(y)), (int(x + w), int(y + h)), 255, 2)

plt.figure(figsize=(10, 10))
plt.imshow(test_image, cmap='gray')
plt.title('Detections')
plt.show(block=True)
