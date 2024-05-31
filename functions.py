import numpy as np


def get_iou(bb1, bb2):
    '''
    input as:
    bbx = [x, y, delta(x), delta(y)]

    Rectangle points, including overlap area rectangle
    1. x_a, y_a
    2. x_a + delta(x_a), y_a
    3. x_a + delta(x_a), y_a + delta(y_a)
    4. x_a, y_a + delta(y_a)

    so the area can be computed as:
    A = h * w, where
    h = delta(x_a)
    w = delta(y_a)

    So rather than point coordinates, we need the difference between two points. For overlap area this will equal to:
    h = abs(max(y_1, y_2) - min(y_1 + delta(y_1), y_2 + delta(y_2)))
    w = abs(max(x_1, x_2) - min(x_1 + delta(x_1), x_2 + delta(x_2)))
    '''

    if (bb1[0] <= bb2[0] and bb1[1] <= bb2[1]) and (bb1[0] + bb1[2] < bb2[0] or bb1[1] + bb1[3] < bb2[1]):
        return 0

    if(bb2[0] <= bb1[0] and bb2[1] <= bb1[1]) and (bb2[0] + bb2[2] < bb1[0] or bb2[1] + bb2[3] < bb1[1]):
        return 0

    # areas
    area_bb1 = bb1[2] * bb1[3]
    area_bb2 = bb2[2] * bb2[3]
    w_overlap = abs(max(bb1[0], bb2[0]) - min(bb1[0] + bb1[2], bb2[0] + bb2[2]))
    h_overlap = abs(max(bb1[1], bb2[1]) - min(bb1[1] + bb1[3], bb2[1] + bb2[3]))
    area_overlap = w_overlap * h_overlap

    return area_overlap / (area_bb2 + area_bb1 - area_overlap)


ar = np.asarray([[320., 128.,  64., 128.],
       [310., 120.,  80., 160.]])

#ar = np.asarray([[310., 120.,  80., 160.],
#       [1000., 1360.,  64., 128.]])

oio = get_iou(ar[0], ar[1])
print(oio)