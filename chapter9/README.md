# Chapter 9 — Activation Functions: What to Put on Each Layer

## The Core Idea

Chapters 1–8 used ReLU everywhere. Chapter 9 asks: **are there better choices?**

Different activation functions shape the output of a layer differently. Choosing the right one for the right layer improves both accuracy and learning speed.

---

## What Makes a Good Activation Function?

Four constraints:

| Constraint | Why it matters |
|---|---|
| **Continuous** | Tiny weight change → tiny output change. Needed for gradient descent to work smoothly. |
| **Monotonic** | Always going in one direction. Non-monotonic creates multiple local minima that are hard to escape. |
| **Nonlinear** | Without it, stacking layers collapses to a single layer (the linear problem from Chapter 6). |
| **Efficient** | Called billions of times during training. Must be fast to compute. |

---

## The Four Activations

### 1. No Activation (raw output)

```python
output = layer.dot(weights)
```

- Use on the **output layer** when predicting a raw number (regression)
- Example: predicting house price, temperature, stock return

---

### 2. Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(output):
    return output * (1 - output)   # output is already sigmoid(x)
```

- Output range: **(0, 1)**
- Use when predicting a **probability** for one thing
- Problem for hidden layers: gradients are small near 0 and 1 → slow learning

---

### 3. Tanh

```python
def tanh(x):
    return np.tanh(x)

def tanh_deriv(output):
    return 1 - output ** 2   # output is already tanh(x)
```

- Output range: **(-1, 1)**
- **Better than sigmoid for hidden layers** — centred at 0, so gradients are larger
- Use on **hidden layers** in most fully-connected networks

---

### 4. Softmax

```python
def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)
```

- Output range: **(0, 1), sum to 1** — a probability distribution
- Use on the **output layer** when classes are **mutually exclusive** (exactly one is correct)
- Delta formula: `(output - true) / (batch_size * num_labels)`

---

## Sigmoid vs Softmax for Classification

Both output values in (0,1). Why prefer softmax?

With sigmoid, each output is predicted **independently** — the network can predict "80% likely class A" and "80% likely class B" at the same time. That's contradictory for mutually exclusive classes.

With softmax, all outputs are **linked** — boosting one automatically lowers the others. Predictions must sum to 1.

Additionally, softmax **shares features**. When the network learns "this looks like a 1", it can simultaneously lower its confidence in "2", "7", etc. With sigmoid, each class fights for features on its own.

---

## Derivative Lookup Table

| Activation | Forward | Derivative (given output) |
|---|---|---|
| Sigmoid | `1 / (1 + exp(-x))` | `output * (1 - output)` |
| Tanh | `tanh(x)` | `1 - output²` |
| ReLU | `max(0, x)` | `output > 0` |
| Softmax | `exp(x) / Σexp(x)` | `(output - true) / (batch × classes)` |

---

## Upgraded MNIST: tanh + softmax

Swapping ReLU for tanh (hidden) and softmax (output) improves test accuracy:

```python
# Forward pass
layer_1 = tanh(np.dot(layer_0, weights_0_1))
layer_1 *= dropout_mask * 2                          # dropout still applies
layer_2 = softmax(np.dot(layer_1, weights_1_2))

# Backward pass
layer_2_delta = (labels - layer_2) / (batch_size * layer_2.shape[0])
layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh_deriv(layer_1)
layer_1_delta *= dropout_mask
```

**Two extra things that matter:**

**Weight initialisation:** tanh is sensitive to large initial weights — use a narrower range:
```python
weights_0_1 = 0.02 * np.random.random((pixels, hidden)) - 0.01   # range: -0.01 to 0.01
```

**Learning rate:** with softmax delta averaging, gradients are smaller — increase alpha to compensate:
```python
alpha = 2   # higher than the 0.001–0.01 used in earlier chapters
```

### Results

```
Chapter 8 (ReLU + sigmoid):            ~82% test accuracy
Chapter 9 (tanh + softmax):            ~87% test accuracy
```

---

## Where to Use Each Activation

| Layer | Task | Use |
|---|---|---|
| Hidden layer | Any | tanh (or ReLU) |
| Output | Predict a number | None (raw) |
| Output | Predict one probability | Sigmoid |
| Output | Predict mutually exclusive class | Softmax |

---

## Key Vocab

| Term | Plain English |
|---|---|
| **Sigmoid** | Squashes any value into (0, 1). Good for probabilities, slow for hidden layers. |
| **Tanh** | Squashes into (−1, 1), centred at 0. Better gradient flow than sigmoid. |
| **Softmax** | Normalises a vector into a probability distribution that sums to 1. |
| **Monotonic** | Always increasing (or always decreasing) — never reverses direction. |
| **Mutually exclusive** | Only one class can be correct at a time (digit recognition, sentiment, etc.). |
| **Weight initialisation** | The starting values of weights. Too large → saturated activations and dead gradients. |

---

## What's Next

Chapter 10 introduces a network that looks at **sequences** — where order matters. The fully connected layers seen so far treat each input as independent. The new architecture in Chapter 10 feeds the output of one step back as input to the next.

---

[← Chapter 8](../chapter8/README.md) | [Back to Main](../README.md) | [Chapter 10 →](../chapter10/README.md)
