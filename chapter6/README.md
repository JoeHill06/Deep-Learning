# Chapter 6 — Backpropagation: Your First Deep Neural Network

## The Core Idea

A single-layer network can only find **direct correlation** between inputs and output.
When no single input correlates with the output (like recognising a cat from pixels), it fails.

The fix: **add a hidden layer** that creates intermediate data which *does* correlate with the output — even when the raw inputs don't.

Getting the hidden layer to learn correctly requires a new technique: **backpropagation**.

---

## The Streetlight Problem

Six streetlight states, each labelled walk or stop.

```
Lights [1,0,1] → STOP
Lights [0,1,1] → WALK
Lights [0,0,1] → STOP
Lights [1,1,1] → WALK
```

The hidden pattern: **only the middle light predicts walk/stop**. The left and right lights are noise.

A 2-layer network eventually finds this. But what if the dataset had *no* direct correlation? That's when you need depth.

---

## How Neural Networks Learn: Up/Down Pressure

Each training example pushes each weight either **up** (toward 1) or **down** (toward 0).

```
Training example [1,0,1] → STOP (0):
  Left weight  → down pressure (input=1, but prediction should be 0)
  Middle weight → no pressure  (input=0, weight is irrelevant)
  Right weight → down pressure (input=1, but prediction should be 0)
```

Over many examples, weights that consistently correlate with the output get pushed up. Weights that don't get pushed down to 0. **Learning = finding correlation through repeated pressure.**

---

## Overfitting

A network can accidentally find a weight configuration that predicts the training data perfectly — without learning the real pattern. It **memorises** instead of **generalises**.

Example: `left_weight=0.5, right_weight=-0.5` predicts 0 for `[1,0,1]` — but these weights would fail on new streetlights.

**The fix:** train on more examples. Each new example disrupts configurations that only work for a subset of data.

> The greatest challenge in deep learning: convincing your network to **generalise** rather than memorise.

---

## Why Stacking Layers Without ReLU Doesn't Work

Two matrix multiplications back-to-back is mathematically equivalent to *one* matrix multiplication:

```
1 * 10 * 2 = 20  ← same as 1 * 20
```

A three-layer linear network has **identical power to a two-layer network**. Stacking without something extra adds zero capability.

---

## The Fix: ReLU (Nonlinearity)

**ReLU** = "Rectified Linear Unit". The simplest nonlinearity:

```python
def relu(x):
    return (x > 0) * x   # if positive, keep it; if negative, set to 0
```

```
Input:  [-2, 0.5, 3, -0.1]
Output: [ 0, 0.5, 3,  0.0]
```

**Why this matters:** A hidden node that gets turned off (set to 0) by ReLU has *zero correlation* to its inputs at that moment. This means the node can be:
- Correlated to input A *only when input B is off*
- Correlated to nothing when conditions aren't right

This is **conditional / sometimes correlation** — impossible without nonlinearity. It's what lets hidden layers create genuinely new information rather than just remixing the inputs.

---

## Backpropagation

Backprop answers: **"How much did each hidden node contribute to the final error?"**

You already know how to compute `delta` for the output node. Backprop propagates that signal backwards through the network.

```
layer_2_delta = layer_2 - true                             # output delta (same as before)

layer_1_delta = layer_2_delta.dot(weights_1_2.T)           # weighted average back through weights
             * relu2deriv(layer_1)                         # zero out nodes that were OFF
```

**Why multiply by `weights_1_2.T`?**
The weights tell you how much each hidden node contributed to the output. Reversing through them tells you how much each hidden node should change.

**Why multiply by `relu2deriv`?**
If relu set a node to 0 in the forward pass, that node contributed *nothing* to the prediction — it gets zero blame.

```python
def relu2deriv(output):
    return output > 0   # 1 if node was ON, 0 if node was OFF
```

---

## The Complete Backpropagation Loop

```python
# FORWARD PASS
layer_0 = input_row
layer_1 = relu(layer_0.dot(weights_0_1))    # hidden layer with relu
layer_2 = layer_1.dot(weights_1_2)          # output (no relu on final layer)

# BACKWARD PASS
layer_2_delta = layer_2 - true
layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

# WEIGHT UPDATES (same outer-product rule from Ch5)
weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
```

This is the complete core of deep learning. Every network you will ever build is a variation of this loop.

---

## Three Types of Gradient Descent

| Type | Updates weights... | Best for |
|---|---|---|
| **Stochastic** | After every single example | Small datasets, noisy but fast |
| **Batch** | After N examples (e.g. 32, 128) | Most common in practice |
| **Full (gradient descent)** | After the whole dataset | Large datasets, stable but slow |

Chapter 6 uses **stochastic** — one weight update per training example.

---

## Why Deep Networks Matter

A shallow network looks for correlation between raw inputs and output.

A deep network creates **intermediate layers** that detect *configurations* of inputs:

```
Raw pixels → [edge detectors] → [shape detectors] → [object detectors] → "cat"
```

No individual pixel correlates with "cat". But a configuration of pixels (cat ear shape) does. A configuration of shapes (pointy ears + whiskers) does even more. Deep networks learn these hierarchies automatically.

---

## Key Vocab

| Term | Plain English |
|---|---|
| **ReLU** | Sets negative node values to 0 — the simplest nonlinearity |
| **relu2deriv** | Returns 1 if node was positive (ON), 0 if not — used in backprop |
| **Backpropagation** | Passing error signal backwards through the network to compute hidden deltas |
| **Nonlinearity** | Any function that breaks the "two matrix mults = one" problem |
| **Conditional correlation** | A hidden node being correlated to inputs *only sometimes* — made possible by ReLU |
| **Overfitting** | Memorising training data instead of learning the underlying pattern |
| **Stochastic gradient descent** | Updating weights after every single training example |

---

## The Code to Memorise

```python
# Forward
layer_1 = relu(layer_0.dot(weights_0_1))
layer_2 = layer_1.dot(weights_1_2)

# Backward
layer_2_delta = layer_2 - true
layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

# Update
weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
```

Memorise this. Every chapter from here on builds on it.

---

[← Chapter 5](../chapter5/README.md) | [Back to Main](../README.md) | [Chapter 7 →](../chapter7/README.md)
