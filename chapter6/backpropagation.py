import numpy as np

# =============================================================================
# CHAPTER 6 — Backpropagation: Your First Deep Neural Network
#
# The streetlight problem: no single input correlates with the output.
# A 2-layer net can't solve it. A 3-layer net with ReLU can.
#
# Dataset: 3 lights (on/off) → walk (1) or stop (0)
# Pattern: middle light = walk, outer lights = stop
# =============================================================================

np.random.seed(1)

# ---------------------------
# DATA
# ---------------------------
streetlights = np.array([
    [1, 0, 1],   # → walk (1)
    [0, 1, 1],   # → walk (1)
    [0, 0, 1],   # → stop (0)
    [1, 1, 1],   # → stop (0)
])
walk_vs_stop = np.array([[1, 1, 0, 0]]).T   # column vector of labels


# ---------------------------
# ACTIVATIONS
# ---------------------------
def relu(x):
    # Turn off any node whose value would be negative.
    # This is what gives the hidden layer "conditional correlation" —
    # it can choose when to pay attention to each input.
    return (x > 0) * x

def relu2deriv(output):
    # Derivative of relu: 1 if the node was ON (>0), else 0.
    # Used in backprop to zero out deltas for nodes that were off.
    return output > 0


# ---------------------------
# HYPERPARAMETERS
# ---------------------------
alpha       = 0.2
hidden_size = 4   # number of nodes in the hidden layer

# Random weights between -1 and 1
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1   # input → hidden
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1   # hidden → output


# =============================================================================
# TRAINING LOOP — Stochastic Gradient Descent + Backpropagation
# =============================================================================
for iteration in range(60):
    layer_2_error = 0

    for i in range(len(streetlights)):

        # ---------------------------------------------------------
        # FORWARD PASS (predict)
        # ---------------------------------------------------------
        layer_0 = streetlights[i:i+1]                    # input row (1×3)
        layer_1 = relu(np.dot(layer_0, weights_0_1))     # hidden layer (1×hidden_size)
        layer_2 = np.dot(layer_1, weights_1_2)           # output prediction (1×1)

        # ---------------------------------------------------------
        # COMPARE (measure error)
        # ---------------------------------------------------------
        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)

        # ---------------------------------------------------------
        # BACKWARD PASS (backpropagation)
        #
        # Step 1: delta at the output node (same as before)
        layer_2_delta = layer_2 - walk_vs_stop[i:i+1]

        # Step 2: propagate delta back to hidden layer
        #   - multiply output delta by weights_1_2 (reversed) → how much each
        #     hidden node contributed to the error
        #   - multiply by relu2deriv → zero out nodes that were OFF (they
        #     contributed nothing to the prediction, so they get no blame)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        # Step 3: compute weight updates (same outer-product rule as Ch5)
        weight_delta_1_2 = layer_1.T.dot(layer_2_delta)
        weight_delta_0_1 = layer_0.T.dot(layer_1_delta)

        # Step 4: update both weight matrices
        weights_1_2 -= alpha * weight_delta_1_2
        weights_0_1 -= alpha * weight_delta_0_1

    if iteration % 10 == 9:
        print(f"Iteration {iteration+1:3d} | Error: {layer_2_error:.8f}")


# ---------------------------
# FINAL PREDICTIONS
# ---------------------------
print("\nFinal predictions:")
for i in range(len(streetlights)):
    layer_0 = streetlights[i:i+1]
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    layer_2 = np.dot(layer_1, weights_1_2)
    label   = "WALK" if walk_vs_stop[i][0] == 1 else "STOP"
    print(f"  Lights {streetlights[i]} → pred={layer_2[0][0]:.4f}  (truth: {label})")
