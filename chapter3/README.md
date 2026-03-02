# Chapter 3 — Forward Propagation: How a Network Makes a Prediction

## The Core Idea

A neural network prediction is just **repeated multiplication and addition**.

That's it. Every network in this chapter — no matter how many inputs, outputs, or layers — boils down to:

```
input × weight = prediction
```

The step of passing data through the network to get a prediction is called **forward propagation**.

---

## What Is a Weight?

A weight is a **volume knob**.

- If weight = 2 → the network doubles the input's impact
- If weight = 0.01 → it shrinks the input's impact to almost nothing
- If weight = 0 → that input is completely ignored
- If weight = -1 → the input pushes the prediction in the opposite direction

The network's "intelligence" is entirely stored in the weight values. Learning = finding the right weights.

---

## The 5 Network Shapes (all in `forward_propagation.py`)

### 1. Single Input → Single Output
The simplest network. One number in, one number out.

```
toes=8.5  →  × weight(0.1)  →  prediction=0.85
```

```python
def neural_network(input, weight):
    return input * weight
```

---

### 2. Multiple Inputs → Single Output
Each input gets its own weight. The final prediction is the **weighted sum** (also called a **dot product**).

```
toes   × 0.1 = 0.85
win%   × 0.2 = 0.13
fans   × 0.0 = 0.00
              ------
               0.98  ← final prediction
```

```python
def neural_network(inputs, weights):
    return inputs.dot(weights)   # dot product = weighted sum
```

**Why does a dot product work?**
It measures *similarity* between your input and your weights. A high dot product means the input strongly matches what the weights are "looking for". High weight on a feature = that feature matters more.

---

### 3. Single Input → Multiple Outputs
One input scaled independently for each output. Three separate volume knobs on the same input.

```
win% × 0.3 = 0.195  ← hurt?
win% × 0.2 = 0.13   ← win?
win% × 0.9 = 0.585  ← sad?
```

```python
def neural_network(input, weights):
    return input * weights   # elementwise multiply
```

Each output is fully independent — there's no connection between them.

---

### 4. Multiple Inputs → Multiple Outputs
Combine both above. You have a **matrix** of weights (a grid).

Each output is its own separate dot product with all the inputs.
Think of it as three networks running in parallel, sharing the same input.

```
weights = [
    [0.1,  0.1, -0.3],   # dot with input → hurt?
    [0.1,  0.2,  0.0],   # dot with input → win?
    [0.0,  1.3,  0.1],   # dot with input → sad?
]
```

```python
def neural_network(inputs, weights):
    return weights.dot(inputs)   # vector-matrix multiplication
```

---

### 5. Stacked Networks — Predicting on Predictions
The output of one network becomes the input to the next.

```
inputs → [weight matrix 1] → hidden values → [weight matrix 2] → final prediction
```

The middle values are called the **hidden layer**. This is what makes a network "deep" — multiple layers of weights stacked on top of each other.

```python
def neural_network(inputs, ih_weights, hp_weights):
    hidden = ih_weights.dot(inputs)    # layer 1
    output = hp_weights.dot(hidden)    # layer 2
    return output
```

---

## Key Vocab

| Term | Plain English |
|---|---|
| **Weight** | A number that controls how much an input influences the prediction |
| **Weighted sum / Dot product** | Multiply each input by its weight, then add them all up |
| **Vector** | A list of numbers (e.g. `[8.5, 0.65, 1.2]`) |
| **Matrix** | A grid of numbers — a list of weight vectors |
| **Elementwise operation** | Pair up values by position and apply the operation (add, multiply, etc.) |
| **Hidden layer** | The layer(s) between input and output in a stacked network |
| **Forward propagation** | The full journey of data through the network to produce a prediction |
| **Activation** | Any number in the network that isn't a weight (inputs, hidden values, outputs) |

---

## NumPy Shape Rule (critical to remember)

When doing `.dot()`, the **inner dimensions must match**:

```
(1, 4).dot(4, 3) → OK  → output shape (1, 3)
(2, 4).dot(4, 3) → OK  → output shape (2, 3)
(2, 4).dot(3, 4) → ERROR — 4 ≠ 3
```

Rule: `(a, b).dot(b, c) = (a, c)` — the middle number must be the same.

---

## What's Missing (the point of Chapter 4)

Right now the weights are just guesses. The network predicts, but it doesn't learn.

Chapter 4 adds the **Compare** and **Learn** steps:
- Measure how wrong the prediction was (error)
- Nudge the weights slightly in the right direction (gradient descent)

That's what transforms a guessing machine into a learning machine.

---

[← Chapter 2](../chapter2/README.md) | [Back to Main](../README.md) | [Chapter 4 →](../chapter4/README.md)
