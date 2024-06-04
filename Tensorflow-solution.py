# Project implementation in sklearn and tensorflow for comparisson
# Windows fix for matplotlib
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from functions import *

# other libs
import numpy as np
import keras
from sklearn.utils import shuffle
from skimage.feature import hog, local_binary_pattern
import cv2
import glob

# epmpty data
x = []
y = []

#hog params
orientation_bins = 16
pixel_window = 32
input_len = int(128 * 64 * orientation_bins / (pixel_window ** 2))

# Try LBP
radius = 2
circle = radius * 8

# Load data - list of numpy arrays
people_x = []
people_y = []
for img in glob.glob("classes\\person\\*.png"):
    im = cv2.imread(img)
    im = cv2.resize(im, (128, 64))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fd = hog(im, orientations=orientation_bins, pixels_per_cell=(pixel_window, pixel_window), cells_per_block=(1, 1))
    lbp = local_binary_pattern(im, circle, radius, 'default')
    people_x.append(fd)
    people_y.append(1)

background_x = []
background_y = []
for img in glob.glob("classes\\background\\*.png"):
    im = cv2.imread(img)
    im = cv2.resize(im, (128, 64))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fd = hog(im, orientations=orientation_bins, pixels_per_cell=(pixel_window, pixel_window), cells_per_block=(1, 1))
    lbp = local_binary_pattern(im, circle, radius, 'default')
    background_x.append(fd)
    background_y.append(0)

# even out datasets
if len(background_y) < len(people_y):
    people_y=people_y[0:len(background_y)]
    people_x=people_x[0:len(background_y)]
else:
    background_y=background_y[0:len(people_y)]
    background_x=background_x[0:len(people_y)]

x = people_x + background_x
y = people_y + background_y

# Shuffle using numpy -> think of etter way later on.
x_train, y_train = shuffle(np.asarray(x), np.asarray(y))
#y_train = keras.utils.to_categorical(y, num_classes=2)

# model
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

model.summary()
optimizer = keras.optimizers.SGD(learning_rate=0.1)
loss = keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.3, callbacks=[SaveBest])
model.load_weights(checkpoint_path)

# Moving window part :D
test_image = cv2.imread("people_2.jpg")
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image_cpy = test_image
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
    idx_to_delete = np.where(iou_tab > 0.05)
    tmp_dets = np.delete(tmp_dets, idx_to_delete, 0)
    tmp_scores = np.delete(tmp_scores, idx_to_delete)

for i, (x, y, w, h) in enumerate(filtered_dets):
    cv2.rectangle(test_image, (int(x), int(y)), (int(x + w), int(y + h)), 255, 2)

plt.figure(figsize=(10, 10))
plt.imshow(test_image, cmap='gray')
plt.title('Detections')
plt.show(block=True)
