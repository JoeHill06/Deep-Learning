"""
Chapter 10 — Intro to Convolutional Neural Networks

Key idea: instead of one big fully-connected layer, use many tiny linear layers
(kernels) that are reused at every position in the image. This slashes the
weight count while letting the network detect the same feature anywhere.

Architecture: 28×28 image → conv layer (3×3, 16 kernels) → tanh → dropout
              → softmax output → 10 classes
"""

import numpy as np
from keras.datasets import mnist

# ─────────────────────────────────────────────
# Activation helpers
# ─────────────────────────────────────────────

def tanh(x):        return np.tanh(x)
def tanh_deriv(o):  return 1 - o ** 2
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# ─────────────────────────────────────────────
# The convolutional helper
# ─────────────────────────────────────────────

def get_image_section(layer, row_from, row_to, col_from, col_to):
    """Pull a (row_to-row_from) × (col_to-col_from) patch out of every image
    in the batch. Returns shape: (batch, 1, patch_h, patch_w)."""
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[:1000].reshape(1000, 28 * 28) / 255.0
labels = np.zeros((1000, 10))
labels[np.arange(1000), y_train[:1000]] = 1

test_images = x_test.reshape(len(x_test), 28 * 28) / 255.0
test_labels = np.zeros((len(y_test), 10))
test_labels[np.arange(len(y_test)), y_test] = 1


# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

np.random.seed(1)
alpha       = 2
iterations  = 300
batch_size  = 128

input_rows  = 28
input_cols  = 28
kernel_rows = 3
kernel_cols = 3
num_kernels = 16

# Each kernel slides over a (28-3) × (28-3) = 25×25 grid of positions.
# With 16 kernels that's 25 × 25 × 16 = 10 000 hidden values.
hidden_size = (input_rows - kernel_rows) * (input_cols - kernel_cols) * num_kernels

print(f"Kernel params : {kernel_rows * kernel_cols} × {num_kernels} = "
      f"{kernel_rows * kernel_cols * num_kernels}")
print(f"FC comparison : {28 * 28} × {hidden_size} = {28 * 28 * hidden_size:,} "
      f"(a fully-connected layer would need this many)")
print(f"Hidden size   : {hidden_size}")
print()


# ─────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────

# kernels: (9, 16) — nine inputs (3×3 patch), 16 output neurons (one per kernel)
kernels    = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01
weights_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

for j in range(iterations):
    correct_cnt = 0

    for i in range(len(images) // batch_size):
        batch_start = i * batch_size
        batch_end   = batch_start + batch_size

        # ── Forward: convolutional layer ──────────────────────────────
        layer_0 = images[batch_start:batch_end].reshape(-1, 28, 28)  # (B, 28, 28)

        # Collect every 3×3 patch from every position in every image
        sects = []
        for row_start in range(input_rows - kernel_rows):       # 0..24
            for col_start in range(input_cols - kernel_cols):   # 0..24
                sect = get_image_section(
                    layer_0,
                    row_start, row_start + kernel_rows,
                    col_start, col_start + kernel_cols
                )
                sects.append(sect)

        # expanded_input: (B, 625, 1, 3, 3) → flattened: (B×625, 9)
        expanded_input  = np.concatenate(sects, axis=1)
        es              = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        # One dot product = ALL kernels at ALL positions in the batch
        kernel_output = flattened_input.dot(kernels)             # (B×625, 16)

        # Reshape back to (B, 625×16) = (B, 10 000)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))

        # Dropout
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2

        # Output
        layer_2 = softmax(layer_1.dot(weights_1_2))             # (B, 10)

        # ── Accuracy ──────────────────────────────────────────────────
        correct_cnt += np.sum(
            np.argmax(layer_2, axis=1) ==
            np.argmax(labels[batch_start:batch_end], axis=1)
        )

        # ── Backward ──────────────────────────────────────────────────
        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / \
                        (batch_size * layer_2.shape[0])

        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh_deriv(layer_1)
        layer_1_delta *= dropout_mask

        # ── Weight updates ────────────────────────────────────────────
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)

        # Propagate delta back through the reshape into kernel space
        l1d_reshape = layer_1_delta.reshape(kernel_output.shape)  # (B×625, 16)
        k_update    = flattened_input.T.dot(l1d_reshape)           # (9, 16)
        kernels    -= alpha * k_update

    # ── Test accuracy (no dropout) ────────────────────────────────────
    if j % 10 == 9:
        test_correct = 0
        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1].reshape(1, 28, 28)
            sects = []
            for row_start in range(input_rows - kernel_rows):
                for col_start in range(input_cols - kernel_cols):
                    sect = get_image_section(
                        layer_0,
                        row_start, row_start + kernel_rows,
                        col_start, col_start + kernel_cols
                    )
                    sects.append(sect)
            expanded_input  = np.concatenate(sects, axis=1)
            es              = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0] * es[1], -1)
            kernel_output   = flattened_input.dot(kernels)
            layer_1         = tanh(kernel_output.reshape(es[0], -1))
            layer_2         = softmax(layer_1.dot(weights_1_2))
            test_correct    += int(np.argmax(layer_2) == np.argmax(test_labels[i]))

        print(
            f"Iteration {j+1:3d} | "
            f"Train-Acc: {correct_cnt / len(images):.3f} | "
            f"Test-Acc: {test_correct / len(test_images):.3f}"
        )
