import math
from cv2 import imread, cvtColor, COLOR_BGR2GRAY
import numpy as np
from math import atan2, sqrt, atan

from skimage.feature import hog

def get_HOG_features(image, pixels_per_cell, bin_count, cells_per_block):

    ### STEP 0 -> declare HOG constants
    PIXELS_PER_CELL = pixels_per_cell
    BIN_COUNT = bin_count
    CELLS_PER_BLOCK = cells_per_block

    ### STEP 1 -> Read image ####
    input_image = image

    ### STEP 2 -> calculate gradient ###
    _1D_Conv = np.asarray([[-1, 0, 1]])
    _filter_step = _1D_Conv.shape[1]

    # predeclare
    Grad_x = np.zeros(input_image.shape)
    Grad_y = np.zeros(input_image.shape)
    angles = np.zeros(input_image.shape, dtype=float)
    Grad = np.zeros(input_image.shape)
    input_padded = np.zeros((input_image.shape[0] + 2, input_image.shape[1] + 2))
    input_padded[1:-1, 1:-1] = input_image

    # X and Y gradients
    for i in range(1, input_padded.shape[0] - 1):
        for j in range(1, input_padded.shape[1] - 1):
            Grad_x[i - 1, j - 1] = np.dot(_1D_Conv, input_padded[i, j - 1:j + 2])[0]

    for i in range(1, input_padded.shape[0] - 1):
        for j in range(1, input_padded.shape[1] - 1):
            Grad_y[i - 1, j - 1] = np.dot(_1D_Conv, input_padded[i - 1:i + 2, j])[0]

    # Magnitude + angle values
    for i in range(Grad.shape[0]):
        for j in range(Grad.shape[1]):
            Grad[i, j] = sqrt((Grad_y[i, j] ** 2) + (Grad_x[i, j] ** 2))
            angle = atan2(Grad_y[i, j], Grad_x[i, j])* 180 / np.pi
            if angle < 0:
                angle = angle + 180
            angles[i, j] = angle
            '''
            if 0 != Grad_x[i, j]:
                angle = atan(Grad_y[i, j] / Grad_x[i, j]) / np.pi * 180
                if angle < 0:
                    angle = 90 + abs(angle)
                angles[i, j] = angle
            else:
                angles[i, j] = 0
            '''

    # Fix border gradient value
    x, y = Grad.shape
    x -= 1
    y -= 1
    '''
    Grad[0, :] = Grad[1, :]
    Grad[-1, :] = Grad[-2, :]
    Grad[:, 0] = Grad[:, 1]
    Grad[:, -1] = Grad[:, -2]

    # Fix corner values
    Grad[0, 0] = (Grad[0, 1] + Grad[1, 0]) / 2
    Grad[x, y] = (Grad[x, y-1] + Grad[x-1, y]) / 2
    Grad[x, 0] = (Grad[x-1, 0] + Grad[x, 1]) / 2
    Grad[0, y] = (Grad[0, y-1] + Grad[1, y]) / 2
    '''
    
    Grad[0, :] = np.zeros(Grad.shape[1])
    Grad[-1, :] = np.zeros(Grad.shape[1])
    Grad[:, 0] = np.zeros(Grad.shape[0])
    Grad[:, -1] = np.zeros(Grad.shape[0])


    # Radian values to degrees
    angles[0, :] = np.zeros(angles.shape[1])
    angles[-1, :] = np.zeros(angles.shape[1])
    angles[:, 0] = np.zeros(angles.shape[0])
    angles[:, -1] = np.zeros(angles.shape[0])

    #angles = angles / math.pi * 180

    ### STEP 3 -> split into patches and calculate histogram
    x_res = int(Grad.shape[0] / PIXELS_PER_CELL)
    y_res = int(Grad.shape[1] / PIXELS_PER_CELL)

    histogram_tensor = np.empty((x_res, y_res), dtype=np.ndarray)

    # analize patches and create histograms
    for i in range(x_res):
        for j in range(y_res):
            gradient_tensor = Grad[i * PIXELS_PER_CELL: PIXELS_PER_CELL * (i + 1),
                                 j * PIXELS_PER_CELL: PIXELS_PER_CELL * (j + 1)]

            angles_tensor = angles[i * PIXELS_PER_CELL: PIXELS_PER_CELL * (i + 1),
                                j * PIXELS_PER_CELL: PIXELS_PER_CELL * (j + 1)]

            # patch histogram
            DEGS = 180 / BIN_COUNT
            histogram = np.zeros(BIN_COUNT, dtype=float)
            cell_A = angles_tensor.flatten()
            cell_M = gradient_tensor.flatten()

            for pixel in range(len(cell_A)):
                # unsigned gradient
                angle = cell_A[pixel]

                angle_bin_value = math.floor(angle / DEGS)
                angle_bin = angle_bin_value % BIN_COUNT
                next_bin = (angle_bin + 1) % BIN_COUNT

                # contribute to bins
                #histogram[angle_bin] += cell_M[pixel] * (DEGS * (angle_bin_value + 1) - angle) / DEGS
                #histogram[next_bin] += cell_M[pixel] * (angle - DEGS * (angle_bin_value)) / DEGS
                histogram[angle_bin] += cell_M[pixel]

            histogram_tensor[i, j] = histogram

    ### STEP 4 -> normalize gradients in blocks ###
    hist_x, hist_y = histogram_tensor.shape
    histogram_normalised = []

    for i in range(hist_x - 1):
        for j in range(hist_y - 1):
            block = histogram_tensor[i:i+CELLS_PER_BLOCK, j:j+CELLS_PER_BLOCK]

            # concatenate patches into blocks
            top = np.hstack((block[0, 0], block[0, 1]))
            bottom = np.hstack((block[1, 0], block[1, 1]))
            block_stacked = np.vstack((top, bottom)).flatten()

            # introduce L2- norm
            e = 0.001
            block_normalised = block_stacked / sqrt((np.linalg.norm(block_stacked)**2) + e**2)

            histogram_normalised.append(block_normalised)


    histogram_normalised = np.asarray(histogram_normalised).flatten()
    return histogram_normalised

#input_image = imread("tst.jpg")
#input_image = cvtColor(input_image, COLOR_BGR2GRAY)
input_image = np.arange(0, 16*16*2, 2)
input_image =np.reshape(input_image, (16, 16))

input_image[:, 3] = np.ones(16) *3
input_image[6, :] = np.ones(16) * 17

#input_image = np.asarray([[1,2,3,4],
#                          [2,5,8,11],
#                         [3,7,11,15],
#                         [4,4,4,4]])


h2 = hog(input_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
h22 = np.reshape(h2, (4, 8))
h1 = get_HOG_features(input_image, 8, 8, 2)

h12 = np.reshape(h1, (4, 8))
