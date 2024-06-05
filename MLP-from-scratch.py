### THIS IS NOT WORKING SADLY :/ ###


import numpy as np
import math
import tensorflow as tf
import tqdm
np.seterr(all='raise') # very useful for any project that you do yourself.

# MSE Loss
def cost_function(y, y_hat):
    eps = 0.0000001
    y = min(y, 1-eps)
    y = max(y, eps)

    y_hat = min(y, 1-eps)
    y_hat = max(y_hat, eps)

    return (1/ (np.size(y)))*np.sum((y_hat  - y ) ** 2)


# MSE Derivative
def cost_function_derivative(y, y_hat):
    return (1 / (np.size(y))) * (-2) * (y_hat - y)


# Cathegorical Crossentropy - UNUSED
def cross_entropy(y, y_hat):
    n = np.size(y_hat)
    tmp = np.zeros(n)
    for ii in range(n):
        tmp[ii] = y_hat[ii] * math.log(y[ii], 10) + ((1- y_hat[ii])* math.log((1 - y[ii]), 10))

    return - sum(tmp)


#  Cathegorical Crossentropy derivative - UNUSED
def cross_entropy_derivative(y, y_hat):
    n = np.size(y_hat)
    tmp = np.zeros(n)
    for ii in range(n):
        tmp[ii] = -1 * ((y_hat[ii] * (1 / y[ii])) + (1 - y_hat[ii])*(1/(1-y[ii])))

    return tmp

# Binary Crossentropy using sigmoid activation function output
def binary_crossentropy(output, target):
    return max(output, 0) - output * target + math.log(1 + math.exp(- abs(output)), np.e)

def binary_crossentropy_derivative(output, target):
    # clip values
    output = max(0.0000001, output)
    return - (target / output) + (1-target) / (1-output)

# Dense with ReLU
class DenseLayer:
    def __init__(self, layer_size, data_size):
        self.neuron_count = layer_size
        self.neuron_tab = []
        self.weight_tab = []
        for _ in range(layer_size):
            # Weights distributed according to centralize normal distribution
            self.weight_tab.append(np.random.normal(0, 1, data_size))

        self.weight_tab = np.asarray(self.weight_tab)

    # Neuron sum
    def input_data(self, input_data):
        out = []

        for data in range(self.neuron_count):
            out.append(np.dot(self.weight_tab[data, :], input_data))

        return out

    # ReLU activation
    def activate_data(self, input):
        out = np.copy(input)

        for ii in range(np.size(out)):
            if out[ii] <= 0:
                out[ii] = 0.0

        return out

    # ReLU derivative
    def derivative(self, input_data, activation="RELU"):
        if activation == "RELU":
            out = []
            for ii in input_data:
                if ii > 0:
                    out.append(1)
                else:
                    out.append(0)

        return np.asarray(out)

    def upload_weights(self, w_tab):
        self.weight_tab = w_tab

    def get_weights(self):
        return self.weight_tab


# Dense layer with sigmoid
class SigmoidLayer:
    def __init__(self, layer_size, data_size):
        self.neuron_count = layer_size
        self.neuron_tab = []
        self.weight_tab = []
        for _ in range(layer_size):
            # Weights distributed according to centralize normal distribution
            self.weight_tab.append((np.random.rand(data_size) * 2) - 1)

        self.weight_tab = np.asarray(self.weight_tab)

    # Neuron sum
    def input_data(self, input_data):
        out = []

        for data in range(self.neuron_count):
            out.append(np.dot(self.weight_tab[data, :], input_data))

        return np.asarray(out)

    # sigmoid activation
    def activate_data(self, input_data):
        out = []
        for ii in input_data:
            # clip
            ii = min(ii, 1e+2)
            ii = max(ii, -1e+2)

            if ii < 0:
                out.append(np.exp(ii) / (1 + np.exp(ii)))
            else:
                out.append(1 / (1 + np.exp(-1 * ii )))

        return np.asarray(out)

    # sigmoid derivative
    def derivative(self, data_after_activation):
        out = []
        for ii in range(np.size(data_after_activation)):
            out.append(data_after_activation[ii] * (1 - data_after_activation[ii]))

        return np.asarray(out)

    # setter
    def upload_weights(self, w_tab):
        self.weight_tab = w_tab

    # getter
    def get_weights(self):
        return self.weight_tab


# Dense with softmax
class Softmax:
    def __init__(self, layer_size, data_size):
        self.neuron_count = layer_size
        self.neuron_tab = []
        self.weight_tab = []
        for _ in range(layer_size):
            # Weights distributed according to centralize normal distribution
            self.weight_tab.append((np.random.rand(data_size) * 2) - 1)

        self.weight_tab = np.asarray(self.weight_tab)

    # Neural sum
    def input_data(self, input_data):
        out = []

        for data in range(self.neuron_count):
            out.append(np.dot(self.weight_tab[data, :], input_data))

        out = np.asarray(out)
        return out

    # Softmax activation offset by C = np.max(input)
    def activate_data(self, data_after_sum):

        e_x = np.exp(data_after_sum - np.max(data_after_sum))
        return np.asarray(e_x / e_x.sum())

    # Softmax derivative offset by C = np.max(input)
    def derivative(self, data_after_sum):
        out = []
        e_x = np.exp(data_after_sum - np.max(data_after_sum))
        for ii in range(np.size(e_x)):
            out.append(e_x[ii] * (e_x.sum() - e_x [ii]) / (e_x.sum()**2))

        return np.asarray(out)

    # Other softmax derivative - Unused
    def derivative_simple(self, input_data):
        out = []
        for ii in input_data:
            out.append(ii * (1 - ii))
        return np.asarray(out)

    # setter
    def upload_weights(self, w_tab):
        self.weight_tab = w_tab

    # getter
    def get_weights(self):
        return self.weight_tab

# Load dataset - precalculated, so that we need not lose time
X_train = np.load("x_train.npy")
Y_train = np.load("y_train.npy")

X_validate = np.load("x_val.npy")
Y_validate = np.load("y_val.npy")

print("Training matrix shape", X_train.shape, Y_train.shape)
print("Testing matrix shape", X_validate.shape, Y_validate.shape)

# MLP start
input_data = len(X_train[0, :])
l1 = DenseLayer(layer_size=64, data_size=input_data)
W1 = l1.get_weights()

l2 = DenseLayer(layer_size=64, data_size=64)
W2 = l2.get_weights()

l3 = DenseLayer(layer_size=32, data_size=64)
W3 = l3.get_weights()

l4 = SigmoidLayer(layer_size=1, data_size=32)
W4= l4.get_weights()

EPOCHS = 20
pic_count = np.shape(X_train)[0]
LEARNING_RATE = 0.1

patience = 4
best_score = np.inf
for _ in range(EPOCHS):
    epoch_score = []

    # tqdm.tqdm zastepuje for_loop, dodajac pasek progresu w konsoli oraz wylicza czas trwania epoki
    for pic in tqdm.tqdm(range(pic_count)):
        # Lambda layer
        in_data = X_train[pic, :]
        pred = Y_train[pic]

        # begin forward pass
        l1_in = l1.input_data(in_data)
        l1_out = l1.activate_data(l1_in)

        l2_in = l2.input_data(l1_out)
        l2_out = l2.activate_data(l2_in)

        l3_in = l3.input_data(l2_out)
        l3_out = l3.activate_data(l3_in)

        l4_in = l4.input_data(l3_out)
        l4_out = l4.activate_data(l4_in)

        # Score
        score = binary_crossentropy(l4_out, pred)

        # END OF FORWARD PASS
        W1_NEW = np.zeros(W1.shape)
        W2_NEW = np.zeros(W2.shape)
        W3_NEW = np.zeros(W3.shape)
        W4_NEW = np.zeros(W4.shape)

        # Begin back prop.
        # Notacja: Pochodna dL/dw wyznaczana jest jako iloczyn h[x]_f * h[x]_2 * h[x]_3, gdzie
        # h[x]_3 to pochodna wejscia do warstwy po wagach = wejscie warstwy poprzedniej
        # h[x]_2 to pochodna wyjscia z warstwy po wejsciu = pochodna funkcji aktywacji
        # h[x]_f jest pochodna funkcji strat po wyjsciu z warstwy = suma pochodnych czastkowych wektora strat po
        # wyjsciu z warstwy. x oznacza numer warstwy.

        # Loss
        # Ze względu na niestabilność pochodnych lokalnych funkcji sigmoid jak i kategorycznej entropii krzyżowej
        # wartości pochodnych będą rozważane wspólnie. Okazuje się, że elementy wprowadzające niestabilność obliczeń
        # ulegają skróceniu przy wyznaczaniu pochodnej funkcji strat względem wagi warstwy sigmoid; wartość ta wynosi
        # różnicę pomiędzy wartością predykowaną przez model, a oczekiwaną.
        combined_derivative = l4_out - pred
        input_derivative = l3_out

        W_4D = np.zeros(np.shape(W4), dtype=np.float64)
        m, n = np.shape(W_4D)
        for i in range(m):
            for j in range(n):
                W_4D[i, j] = combined_derivative[i] * input_derivative[j]

        W4_NEW = W4 - LEARNING_RATE * W_4D

        # WARSTWA trzecia
        h3_3 = l2_out
        h3_2 = l3.derivative(l3_out)
        h3_1 = np.zeros((m, n), dtype=np.float64)

        for i in range(m):
            for j in range(n):
                h3_1[i, j] = W4[i, j] * combined_derivative[i]

        h3_f = np.zeros(np.size(h3_2), dtype=np.float64)
        h3_f = np.sum(h3_1, axis=0) # sum of column values
        #for i in range(m):
        #    h3_f[i] = np.sum(h3_1[:, i], axis=0)

        W3_D = np.zeros((np.shape(W3)), dtype=np.float64)
        m, n = np.shape(W3)

        for i in range(m):
            for j in range(n):
                W3_D[i, j] = h3_f[i] * h3_2[i] * h3_3[j]

        W3_NEW = W3 - LEARNING_RATE * W3_D
        # B3 = B3 - LEARNING_RATE * np.dot(h3_f, h3_2)

        # WARSTWA druga
        h2_3 = l1_out
        h2_2 = l1.derivative(l2_out)
        h2_1 = np.zeros((m, n), dtype=np.float64)

        for i in range(m):
            for j in range(n):
                h2_1[i, j] = W3[i, j] * h3_2[i] * h3_f[i]

        h2_f = np.zeros(np.size(h2_2), dtype=np.float64)
        for i in range(m):
            h2_f[i] = np.sum(h2_1[i, :])

        m, n = np.shape(W2)
        W2_D = np.zeros((np.shape(W2)), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                W2_D[i, j] = h2_f[i] * h2_2[i] * h2_3[j]

        W2_NEW = W2 - LEARNING_RATE * W2_D
        # B2 = B2 - LEARNING_RATE * np.dot(h2_f, h2_2)

        # WARSTWA pierwsza
        h1_3 = in_data
        h1_2 = l1.derivative(l1_out)
        h1_1 = np.zeros((m, n), dtype=np.float64)

        for i in range(m):
            for j in range(n):
                h1_1[i, j] = W2[i, j] * h2_2[i] * h2_f[i]

        h1_f = np.zeros(np.size(h2_2), dtype=np.float64)
        for i in range(m):
            h1_f[i] = np.sum(h1_1[i, :])

        m, n = np.shape(W1)
        W1_D = np.zeros((np.shape(W1)), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                W1_D[i, j] = h1_f[i] * h1_2[i] * h1_3[j]

        W1_NEW = W1 - LEARNING_RATE * W1_D
        # B1 = B1 - LEARNING_RATE * np.dot(h1_f, h1_2)

        W1 = W1_NEW
        W2 = W2_NEW
        W3 = W3_NEW
        W4 = W4_NEW

        l1.upload_weights(W1)
        l2.upload_weights(W2)
        l3.upload_weights(W3)
        l4.upload_weights(W4)

    # forward pass dla zbioru testowego celem okreslenia loss epoki
    for pic in range(np.shape(X_validate)[0]):
        A = X_validate[pic,:]
        pred = Y_validate[pic]

        l1_in = l1.input_data(A)
        l1_out = l1.activate_data(l1_in)

        l2_in = l2.input_data(l1_out)
        l2_out = l2.activate_data(l2_in)

        l3_in = l3.input_data(l2_out)
        l3_out = l3.activate_data(l3_in)

        l4_in = l4.input_data(l3_out)
        l4_out = l4.activate_data(l4_in)

        # Score
        score = binary_crossentropy(l4_out, pred)
        epoch_score.append(score)

    epoch_score = np.asarray(epoch_score, dtype=float)
    epoch_score = np.sum(epoch_score) / np.size(epoch_score)

    if (best_score > epoch_score):
        best_score = epoch_score
        patience = 8
        np.save("W1_best", W1)
        np.save("W2_best", W2)
        np.save("W3_best", W3)
        np.save("W4_best", W4)
    else:
        patience -= 1
        print("Loss value did not improve, patience flag: ", patience)

    print(epoch_score, _)
    if patience == 0:
        break


predict1 = []
for pic in range(np.shape(X_validate)[0]):
    A = X_validate[pic, :, :].flatten()
    pred = Y_validate[pic, :]

    l1_in = l1.input_data(A)
    l1_out = l1.activate_data(l1_in)

    l2_in = l2.input_data(l1_out)
    l2_out = l2.activate_data(l2_in)

    l3_in = l3.input_data(l2_out)
    l3_out = l3.activate_data(l3_in)

    l4_in = l4.input_data(l3_out)
    l4_out = l4.activate_data(l4_in)
    predict1.append(l4_out)


predict1 = np.asarray(predict1)

p1 = np.argmax(predict1, axis = -1)
r1 = np.argmax(Y_validate, axis = -1)

matrix = np.zeros((2, 2), dtype=int)
for ii in range(np.size(p1)):
    matrix[r1[ii],p1[ii]] += 1

acc = np.trace(matrix)
acc = acc / np.sum(matrix)

accuracy = []
precision = []
recall = []

for ii in range(2):
    recall.append(matrix[ii, ii] / np.sum(matrix[:, ii]))
    precision.append(matrix[ii, ii] /np.sum(matrix[ii, :]))
    accuracy.append((np.sum(matrix) - np.sum(matrix[ii, :])- np.sum(matrix[:, ii]) + matrix[ii, ii] )/ np.sum(matrix))
