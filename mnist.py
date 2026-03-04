# previously (ch9):  I:290 Train-Acc:0.94  Test-Acc:0.870
#    (ch10):  I:299 Train-Acc:~0.82  Test-Acc:~0.877 would increase more with more test images

import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000])

one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def tanh(x):        return np.tanh(x)
def tanh_deriv(o):  return 1 - o ** 2

def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)

# ── Hyperparameters ───────────────────────────────────────────────────────
alpha      = 2
iterations = 300
batch_size = 128

input_rows  = 28
input_cols  = 28
kernel_rows = 3
kernel_cols = 3
num_kernels = 16

# Conv output: (28-3) × (28-3) = 25×25 positions, 16 kernels each
hidden_size = (input_rows - kernel_rows) * (input_cols - kernel_cols) * num_kernels

num_labels = 10

# ── Weights ───────────────────────────────────────────────────────────────
kernels     = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01
weights_1_2 = 0.2  * np.random.random((hidden_size, num_labels)) - 0.1

# ── Training loop ─────────────────────────────────────────────────────────
for j in range(iterations):
    error, correct_cnt = (0.0, 0)

    for i in range(len(images) // batch_size):
        batch_start = i * batch_size
        batch_end   = batch_start + batch_size

        # ── Forward ───────────────────────────────────────────────────────
        layer_0 = images[batch_start:batch_end].reshape(-1, 28, 28)

        sects = []
        for row_start in range(input_rows - kernel_rows):
            for col_start in range(input_cols - kernel_cols):
                sects.append(get_image_section(
                    layer_0,
                    row_start, row_start + kernel_rows,
                    col_start, col_start + kernel_cols))

        expanded_input  = np.concatenate(sects, axis=1)
        es              = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1       = tanh(kernel_output.reshape(es[0], -1))

        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2

        layer_2 = softmax(layer_1.dot(weights_1_2))

        # ── Accuracy ──────────────────────────────────────────────────────
        error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k+1])) == \
                           np.argmax(labels[batch_start+k:batch_start+k+1])

        # ── Backward ──────────────────────────────────────────────────────
        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / \
                        (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh_deriv(layer_1)
        layer_1_delta *= dropout_mask

        # ── Weight updates ────────────────────────────────────────────────
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)

        l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
        k_update    = flattened_input.T.dot(l1d_reshape)
        kernels    -= alpha * k_update

    # ── Test evaluation ───────────────────────────────────────────────────
    if j % 10 == 0:
        test_error     = 0.0
        test_correct_cnt = 0

        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1].reshape(-1, 28, 28)

            sects = []
            for row_start in range(input_rows - kernel_rows):
                for col_start in range(input_cols - kernel_cols):
                    sects.append(get_image_section(
                        layer_0,
                        row_start, row_start + kernel_rows,
                        col_start, col_start + kernel_cols))

            expanded_input  = np.concatenate(sects, axis=1)
            es              = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0] * es[1], -1)

            kernel_output = flattened_input.dot(kernels)
            layer_1       = tanh(kernel_output.reshape(es[0], -1))
            layer_2       = softmax(layer_1.dot(weights_1_2))

            test_error       += np.sum((test_labels[i:i+1] - layer_2) ** 2)
            test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

        sys.stdout.write("\rI:" + str(j) +
            " Train-Err:" + str(error / float(len(images)))[0:5] +
            " Train-Acc:" + str(correct_cnt / float(len(images)))[0:5] +
            " Test-Err:"  + str(test_error / float(len(test_images)))[0:5] +
            " Test-Acc:"  + str(test_correct_cnt / float(len(test_images)))[0:5])
        sys.stdout.flush()
