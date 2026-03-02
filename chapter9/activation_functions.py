"""
Chapter 9 — Activation Functions

Covers: sigmoid, tanh, softmax, and upgrading MNIST with better activations.
"""

import numpy as np
from keras.datasets import mnist

# ─────────────────────────────────────────────
# The four activation functions
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(output):
    # output is already sigmoid(x)
    return output * (1 - output)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(output):
    # output is already tanh(x)
    return 1 - output ** 2

def softmax(x):
    # subtract max for numerical stability (same result, avoids overflow)
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def relu(x):
    return (x > 0) * x

def relu_deriv(output):
    return output > 0


# ─────────────────────────────────────────────
# Demo: activation shapes
# ─────────────────────────────────────────────

print("=== Activation function outputs ===")
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input:    {x}")
print(f"sigmoid:  {np.round(sigmoid(x), 3)}")
print(f"tanh:     {np.round(np.tanh(x), 3)}")
print(f"relu:     {np.round(relu(x), 3)}")

print()
print("=== Softmax example (multi-class output) ===")
logits = np.array([[2.0, 1.0, 0.5]])  # raw scores for 3 classes
probs = softmax(logits)
print(f"Logits:        {logits}")
print(f"Softmax probs: {np.round(probs, 3)}")
print(f"Sum:           {probs.sum():.4f}")   # always sums to 1


# ─────────────────────────────────────────────
# Upgraded MNIST: tanh hidden + softmax output
# ─────────────────────────────────────────────

print()
print("=== Upgraded MNIST: tanh + softmax + dropout + batching ===")

# --- load and prepare data ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()

images  = x_train[:1000].reshape(1000, 784) / 255.0
labels  = np.zeros((1000, 10))
labels[np.arange(1000), y_train[:1000]] = 1

test_images = x_test[:1000].reshape(1000, 784) / 255.0
test_labels = np.zeros((1000, 10))
test_labels[np.arange(1000), y_test[:1000]] = 1

# --- hyperparameters ---
np.random.seed(1)
alpha          = 2
iterations     = 300
hidden_size    = 100
batch_size     = 100
pixels         = 784
num_labels     = 10

# --- weight initialisation (narrow for tanh) ---
weights_0_1 = 0.02 * np.random.random((pixels, hidden_size)) - 0.01
weights_1_2 = 0.2  * np.random.random((hidden_size, num_labels)) - 0.1

# --- training loop ---
for j in range(iterations):
    train_error = 0
    train_correct = 0

    for batch_start in range(0, len(images), batch_size):
        batch_end = batch_start + batch_size

        # --- forward pass ---
        layer_0 = images[batch_start:batch_end]               # (100, 784)
        layer_1 = tanh(np.dot(layer_0, weights_0_1))          # (100, 100)

        # dropout on hidden layer
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2

        layer_2 = softmax(np.dot(layer_1, weights_1_2))       # (100, 10)

        # --- error and accuracy ---
        batch_labels = labels[batch_start:batch_end]
        train_error += np.sum((batch_labels - layer_2) ** 2)
        train_correct += np.sum(
            np.argmax(layer_2, axis=1) == np.argmax(batch_labels, axis=1)
        )

        # --- backward pass ---
        # softmax delta: (pred - true) averaged over batch × output size
        layer_2_delta = (batch_labels - layer_2) / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh_deriv(layer_1)
        layer_1_delta *= dropout_mask   # only update nodes that were on

        # --- weight updates ---
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if j % 10 == 9:
        # test accuracy (no dropout)
        test_layer_1 = tanh(np.dot(test_images, weights_0_1))
        test_layer_2 = softmax(np.dot(test_layer_1, weights_1_2))
        test_correct = np.sum(
            np.argmax(test_layer_2, axis=1) == np.argmax(test_labels, axis=1)
        )
        print(
            f"Iteration {j+1:3d} | "
            f"Train-Acc: {train_correct / len(images):.3f} | "
            f"Test-Acc: {test_correct / len(test_images):.3f}"
        )
