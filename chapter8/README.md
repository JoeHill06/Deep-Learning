# Chapter 8 — Regularization and Batching: Learning Signal, Ignoring Noise

## The Core Idea

A network that trains to 100% accuracy on training data and only 70% on new data hasn't learned — it has **memorised**. This is overfitting, and it's the biggest practical problem in deep learning.

This chapter covers the two most universally used solutions:
1. **Dropout** — randomly turn off neurons during training
2. **Mini-batch gradient descent** — update weights on averaged batches, not single examples

---

## Overfitting

### What it looks like

```
Iteration  10 | Train-Acc: 0.901 | Test-Acc: 0.811   ← test still climbing
Iteration  20 | Train-Acc: 0.930 | Test-Acc: 0.811   ← test plateaus
Iteration  50 | Train-Acc: 0.966 | Test-Acc: 0.798   ← test starts falling
Iteration 349 | Train-Acc: 1.000 | Test-Acc: 0.707   ← train perfect, test terrible
```

Train accuracy keeps climbing while test accuracy **falls**. The network is fitting to noise in the training data.

### The fork mold analogy

Imagine pressing forks into wet clay to make a mould. Press one fork a few times → rough, general fork shape that works for most forks. Press the same fork 1000 times → perfect imprint of *that specific fork* — a four-pronged fork won't fit anymore.

Neural networks do the same thing. The more you train, the more they fit to the fine details (noise) of the training data instead of the general pattern (signal).

### Signal vs. Noise

| Signal | Noise |
|---|---|
| The shape of a digit | Random pixel variations between scans |
| The edges that define a dog | The pillow in the background |
| Consistent patterns across all examples | Unique quirks of individual examples |

Regularization = methods that force the network to learn signal and ignore noise.

---

## Solution 1: Early Stopping

The simplest regularization. Watch your test accuracy during training and **stop when it peaks**.

```
Test-Acc peaks at iteration 20 → stop there, don't train to iteration 349
```

Cheap and effective in a pinch. Downside: you need to evaluate the test set constantly.

---

## Solution 2: Dropout

**Randomly turn off 50% of hidden nodes on each training step.**

```python
dropout_mask  = np.random.randint(2, size=layer_1.shape)  # 1s and 0s
layer_1      *= dropout_mask * 2                           # kill 50%, scale up the rest
```

Apply the same mask during backpropagation so turned-off nodes don't get updated:
```python
layer_1_delta *= dropout_mask
```

At **test time**, use the full network with no dropout — just remove those two lines.

### Why multiply by 2?

When half the nodes are off, the weighted sum feeding into layer_2 is halved. This confuses layer_2 at test time when all nodes are suddenly back on. Multiplying by `1 / keep_rate` (here `1 / 0.5 = 2`) keeps the expected activation level the same between training and testing.

### Why dropout works

**Reason 1 — Forces the network to act small:**
Big networks overfit because they have room to memorise fine details. Turning off random nodes makes the active network smaller — small networks can only learn broad features.

**Reason 2 — Ensembling:**
Each training step uses a different random subnetwork. You're effectively training millions of slightly different small networks and averaging their results. Each overfits to *different* noise, so the noise cancels out. The signal — what they all learned in common — remains.

```
100 networks × (overfit to different noise) → average → only signal survives
```

### Results

```
Without dropout: peaks 81%, falls to 70.7% by end of training
With dropout:    peaks 82.4%, stays at 81.8% by end of training
```

---

## Solution 3: Mini-batch Gradient Descent

Instead of updating weights after every single example, accumulate gradients over a **batch** of examples and update once with their average.

```python
batch_size = 100

layer_0 = images[batch_start:batch_end]          # (100 × 784) — whole batch at once
layer_1 = relu(np.dot(layer_0, weights_0_1))     # (100 × hidden)
layer_2 = np.dot(layer_1, weights_1_2)           # (100 × 10)

# Average the gradient over the batch
layer_2_delta = (labels[batch_start:batch_end] - layer_2) / batch_size
```

### Why batching helps

| Problem with single examples | How batching fixes it |
|---|---|
| Gradient is very noisy (one example = one opinion) | Averaging 100 examples smooths out the noise |
| Must use small alpha to avoid diverging | Averaged gradient is stable → can use bigger alpha |
| Slow — sequential computation | Whole batch = one vectorised dot product → much faster |

The compass analogy: looking at one noisy compass reading and running 2 miles gets you lost. Averaging 100 readings and then running 2 miles gets you close to the right direction.

**Typical batch sizes:** 8, 16, 32, 64, 128, 256. Tune `batch_size` and `alpha` together.

---

## Three Versions Side by Side

| Version | Train Acc | Test Acc | Notes |
|---|---|---|---|
| No regularization | 100% | 70.7% | Classic overfitting |
| + Dropout | ~90% | ~82% | Slows train, boosts test |
| + Dropout + Batching | ~85% | ~80% | Smoother, faster training |

Note: dropout intentionally hurts train accuracy — this is expected and correct.

---

## Key Vocab

| Term | Plain English |
|---|---|
| **Overfitting** | Memorising training data; failing on new data |
| **Generalisation** | Learning the underlying pattern that works on unseen data |
| **Regularisation** | Any method that discourages overfitting |
| **Early stopping** | Stop training when test accuracy peaks |
| **Dropout** | Randomly zeroing hidden nodes during training |
| **Dropout mask** | A random binary matrix (1s and 0s) applied to a layer |
| **Mini-batch** | A fixed-size subset of training examples used per weight update |
| **Keep rate** | Fraction of nodes left ON during dropout (0.5 = 50% on) |

---

## The Three-Line Dropout Change

```python
# After computing layer_1, add these three lines:
dropout_mask  = np.random.randint(2, size=layer_1.shape)
layer_1      *= dropout_mask * 2

# And this one line in backprop:
layer_1_delta *= dropout_mask
```

That's the entire implementation. Four lines of code, significant improvement in generalisation.

---

## What's Next

Chapter 9 introduces a completely new type of neural network architecture — one designed specifically for a type of data where the position and order of information matters. The fully connected layers from Chapters 3–8 treat every input independently; the new architecture in Chapter 9 does not.

---

[← Chapter 7](../chapter7/README.md) | [Back to Main](../README.md) | [Chapter 9 →](../chapter9/README.md)
