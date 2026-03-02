import numpy as np

# =============================================================================
# CHAPTER 3 — Forward Propagation
# All the ways a neural network can make a prediction.
# Every example here is PREDICT only — no learning yet (that's Chapter 4).
# =============================================================================


# -----------------------------------------------------------------------------
# 1. SINGLE INPUT → SINGLE OUTPUT
#    The simplest possible network: one number in, one number out.
#    The weight is a "volume knob" — it scales the input up or down.
# -----------------------------------------------------------------------------

weight = 0.1

def single_in_single_out(input, weight):
    return input * weight

number_of_toes = [8.5, 9.5, 10.0, 9.0]
pred = single_in_single_out(number_of_toes[0], weight)
print(f"1) Single input → single output: {pred}")
# 8.5 * 0.1 = 0.85


# -----------------------------------------------------------------------------
# 2. MULTIPLE INPUTS → SINGLE OUTPUT
#    Each input gets its own weight (volume knob).
#    The final prediction is the weighted sum (dot product) of all inputs.
#    Dot product = similarity between input and weights.
# -----------------------------------------------------------------------------

weights = np.array([0.1, 0.2, 0.0])   # weights for: toes, win%, fans

def multi_in_single_out(inputs, weights):
    # Multiply each input by its weight, then sum everything up
    return inputs.dot(weights)

toes  = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input_game1 = np.array([toes[0], wlrec[0], nfans[0]])
pred = multi_in_single_out(input_game1, weights)
print(f"2) Multiple inputs → single output: {pred:.4f}")
# (8.5*0.1) + (0.65*0.2) + (1.2*0.0) = 0.85 + 0.13 + 0.0 = 0.98


# -----------------------------------------------------------------------------
# 3. SINGLE INPUT → MULTIPLE OUTPUTS
#    One input, but we want several predictions.
#    Each output has its own weight — they are completely independent.
#    This is just elementwise multiplication (input * each weight separately).
# -----------------------------------------------------------------------------

weights_multi_out = np.array([0.3, 0.2, 0.9])  # hurt?, win?, sad?

def single_in_multi_out(input, weights):
    # Scale the same input by each output's weight independently
    return input * weights

pred = single_in_multi_out(wlrec[0], weights_multi_out)
print(f"3) Single input → multiple outputs: {pred}")
# [0.65*0.3, 0.65*0.2, 0.65*0.9] = [0.195, 0.13, 0.585]


# -----------------------------------------------------------------------------
# 4. MULTIPLE INPUTS → MULTIPLE OUTPUTS
#    Combine both above: a grid of weights (a matrix).
#    Each output is its own independent dot product with all the inputs.
#    weights shape: (num_outputs, num_inputs)
# -----------------------------------------------------------------------------

#                    toes   win%   fans
weights_matrix = np.array([
    [ 0.1,  0.1, -0.3],   # → hurt?
    [ 0.1,  0.2,  0.0],   # → win?
    [ 0.0,  1.3,  0.1],   # → sad?
])

def multi_in_multi_out(inputs, weights):
    # For each output row, do a dot product with the input vector
    return weights.dot(inputs)

pred = multi_in_multi_out(input_game1, weights_matrix)
print(f"4) Multiple inputs → multiple outputs: {np.round(pred, 3)}")
# hurt=0.555, win=0.98, sad=0.965


# -----------------------------------------------------------------------------
# 5. STACKED NETWORKS — "Predicting on Predictions"
#    The output of one network becomes the input to the next.
#    This is the foundation of DEEP learning.
#    Layer between input and output is called a HIDDEN LAYER.
#    The full forward pass is called FORWARD PROPAGATION.
# -----------------------------------------------------------------------------

#                        toes   win%   fans
ih_weights = np.array([
    [ 0.1,  0.2, -0.1],  # → hidden[0]
    [-0.1,  0.1,  0.9],  # → hidden[1]
    [ 0.1,  0.4,  0.1],  # → hidden[2]
])

#                        hid0  hid1  hid2
hp_weights = np.array([
    [ 0.3,  1.1, -0.3],  # → hurt?
    [ 0.1,  0.2,  0.0],  # → win?
    [ 0.0,  1.3,  0.1],  # → sad?
])

def stacked_network(inputs, ih_weights, hp_weights):
    hidden = ih_weights.dot(inputs)   # input → hidden layer
    output = hp_weights.dot(hidden)   # hidden → output layer
    return output

pred = stacked_network(input_game1, ih_weights, hp_weights)
print(f"5) Stacked network (forward propagation): {np.round(pred, 3)}")
