import sys
import numpy as np
from keras.datasets import mnist

# =============================================================================
# CHAPTER 8 — Regularization and Batching
# The problem: without regularization a network memorises training data
# and fails badly on new data (overfitting).
# The solutions: Dropout + Mini-batch gradient descent.
# =============================================================================

# ---------------------------
# DATA
# ---------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:1000].reshape(1000, 28*28) / 255
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)

relu       = lambda x: (x >= 0) * x
relu2deriv = lambda x:  x >= 0


# =============================================================================
# VERSION 1: No regularization — overfits badly
# Train acc → 100%, test acc falls to ~70% as training continues
# =============================================================================
print("=== VERSION 1: No Regularization ===")

alpha, iterations, hidden_size = 0.005, 60, 40
pixels_per_image, num_labels    = 784, 10

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels))       - 0.1

for j in range(iterations):
    error, correct_cnt = 0.0, 0
    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error       += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = labels[i:i+1] - layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if j % 20 == 0:
        test_error, test_correct = 0.0, 0
        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)
            test_error   += np.sum((test_labels[i:i+1] - layer_2) ** 2)
            test_correct += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

        print(f"  I:{j:3d} | Train-Acc:{correct_cnt/len(images):.3f} "
              f"| Test-Acc:{test_correct/len(test_images):.3f}")


# =============================================================================
# VERSION 2: Dropout
# Randomly turn off 50% of hidden nodes each forward pass during training.
# Multiply remaining activations by 2 to keep the expected total unchanged.
# Apply the same mask to layer_1_delta during backprop.
# Result: test acc peaks higher AND stays higher through training.
# =============================================================================
print("\n=== VERSION 2: Dropout (50%) ===")

np.random.seed(1)
alpha, iterations, hidden_size = 0.005, 60, 100

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels))       - 0.1

for j in range(iterations):
    error, correct_cnt = 0.0, 0
    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))

        # --- DROPOUT ---
        # Random binary mask: ~50% of nodes set to 0 (turned off)
        dropout_mask  = np.random.randint(2, size=layer_1.shape)
        layer_1      *= dropout_mask * 2   # *2 because we halved the active nodes

        layer_2 = np.dot(layer_1, weights_1_2)

        error       += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = labels[i:i+1] - layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        layer_1_delta *= dropout_mask   # don't update weights of OFF nodes

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if j % 20 == 0:
        test_error, test_correct = 0.0, 0
        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))   # NO dropout at test time
            layer_2 = np.dot(layer_1, weights_1_2)
            test_error   += np.sum((test_labels[i:i+1] - layer_2) ** 2)
            test_correct += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

        print(f"  I:{j:3d} | Train-Acc:{correct_cnt/len(images):.3f} "
              f"| Test-Acc:{test_correct/len(test_images):.3f}")


# =============================================================================
# VERSION 3: Dropout + Mini-batch gradient descent
# Instead of updating weights after every example, accumulate gradients
# over batch_size examples then update once (averaging the gradient).
# Benefits:
#   - Smoother, more stable weight updates
#   - Larger alpha is safe (averaging reduces gradient noise)
#   - Faster: vectorised dot products over the whole batch at once
# =============================================================================
print("\n=== VERSION 3: Dropout + Mini-batch (batch_size=100) ===")
# Note: batch GD uses a much smaller alpha (0.001 vs 0.005) because the
# averaged gradient is already stable. It needs more iterations to converge —
# the book runs 300. Increase iterations here to see the full benefit.

np.random.seed(1)
batch_size              = 100
alpha, iterations       = 0.01, 300
pixels_per_image        = 784
num_labels, hidden_size = 10, 100

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels))       - 0.1

for j in range(iterations):
    error, correct_cnt = 0.0, 0

    for i in range(int(len(images) / batch_size)):
        batch_start = i * batch_size
        batch_end   = (i + 1) * batch_size

        # Whole batch through at once — much faster than one-at-a-time
        layer_0 = images[batch_start:batch_end]           # (100 × 784)
        layer_1 = relu(np.dot(layer_0, weights_0_1))      # (100 × hidden)

        dropout_mask  = np.random.randint(2, size=layer_1.shape)
        layer_1      *= dropout_mask * 2

        layer_2 = np.dot(layer_1, weights_1_2)            # (100 × 10)

        error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
        for k in range(batch_size):
            correct_cnt += int(
                np.argmax(layer_2[k:k+1]) ==
                np.argmax(labels[batch_start+k:batch_start+k+1])
            )

        # /batch_size averages the gradient over the batch
        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / batch_size
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if j % 20 == 0:
        test_error, test_correct = 0.0, 0
        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)
            test_error   += np.sum((test_labels[i:i+1] - layer_2) ** 2)
            test_correct += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

        print(f"  I:{j:3d} | Train-Acc:{correct_cnt/len(images):.3f} "
              f"| Test-Acc:{test_correct/len(test_images):.3f}")
