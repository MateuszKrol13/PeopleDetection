import numpy as np
import cv2
from HOG import get_HOG_features
from functions import get_iou
import tqdm

from sklearn import svm

import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

class SVM_classifier():
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in tqdm.tqdm(range(self.no_of_iterations)):
            self.update_weights()

    def update_weights(self):
        y_label = np.where(self.Y <= 0, -1, 1)

        for index, x_i in enumerate(self.X):
            if y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1:
                dw = 2 * self.lambda_parameter * self.w
                db = 0

            else:
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) - self.b

    def predict_class_only(self, X):
        # this is for confusion matrix and pure classification
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        return y_hat # returns classes -> 0 or 1

# Load pregenerated HOG-from-scratch features -> save time
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

x_validate = np.load("x_val.npy")
y_validate = np.load("y_val.npy")

orientation_bins = 16
pixel_window = 16

model = SVM_classifier(learning_rate=0.001, no_of_iterations=2000, lambda_parameter=0.01)
model.fit(x_train, y_train)

## SVM sklearn
# model = svm.LinearSVC()
# model.fit(x_train, y_train)

########################################################################################################################
########################################## Moving Window ###############################################################

test_image = cv2.imread("people_4.jpg")
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
    for i in tqdm.tqdm(range(count_h)):
        for j in range(count_v):
            horizontal = int(i * step)
            vertical = int(j * step)

            sub_img = curr_img[horizontal:horizontal+sh, vertical:vertical+sv]
            ft = get_HOG_features(sub_img, pixels_per_cell=pixel_window, bin_count=orientation_bins, cells_per_block=2)
            ft_container.append(ft)

    features_map = np.asarray(ft_container)
    out_map = model.predict_with_values(features_map)
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


test_image = cv2.imread("people_4.jpg")
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
    idx_to_delete = np.where(iou_tab > 0.05)
    tmp_dets = np.delete(tmp_dets, idx_to_delete, 0)
    tmp_scores = np.delete(tmp_scores, idx_to_delete)

for i, (x, y, w, h) in enumerate(filtered_dets):
    cv2.rectangle(test_image, (int(x), int(y)), (int(x + w), int(y + h)), 255, 2)

plt.figure(figsize=(10, 10))
plt.imshow(test_image, cmap='gray')
plt.title('Detections')
plt.show(block=True)
