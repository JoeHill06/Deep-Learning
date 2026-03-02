# Chapter 10 — Convolutional Neural Networks: Learning Edges and Corners

## The Core Idea

Overfitting happens when there are too many weights relative to the amount of training data. Regularization (Chapter 8) fights this after the fact. A better solution is **structure** — design the network so it needs fewer weights to begin with.

**The structure trick:**
> When a neural network needs to use the same idea in multiple places, use the same weights in both places. Those weights learn from more examples → better generalisation.

Convolutions are the most famous application of this trick.

---

## The Problem with Fully-Connected Layers on Images

A fully-connected MNIST network has `784 × hidden_size` weights in the first layer. Every input pixel has its own dedicated weight to every hidden neuron. That means:

- A pixel in the top-left and a pixel in the bottom-right each have completely independent weights
- If a network learns "a horizontal edge at position (5,5) means something", it has **no way** to apply that same knowledge at position (15,5)
- Each location must independently re-learn every low-level feature

---

## Convolutional Kernels

Instead of one big matrix, use **many tiny matrices** (kernels). Each kernel:
- Has only `kernel_rows × kernel_cols` inputs (e.g., 3×3 = 9 weights)
- Produces a single output
- **Slides across the entire image** — applied at every position

```
28×28 image, 3×3 kernel → kernel applied at (28−3) × (28−3) = 625 positions
```

Because the same weights are used at every position, the kernel can learn "I detect horizontal edges" — and that single piece of intelligence is applied everywhere in the image simultaneously.

---

## Multiple Kernels

You usually have many kernels, each learning a different low-level feature:

```
Kernel 1: detects horizontal edges
Kernel 2: detects diagonal lines (↗)
Kernel 3: detects corners
Kernel 4: detects curves
...
Kernel 16: detects some other pattern
```

Each kernel scans the whole image independently and produces its own 25×25 output grid. The result is 16 different "feature maps" of the image.

---

## Pooling

After each kernel scans the image you have 16 output grids (6×6 or 25×25 depending on image size). You need to combine them into a single value per position.

| Method | How |
|---|---|
| Sum pooling | Add all kernel outputs at each position |
| Mean pooling | Average all kernel outputs at each position |
| **Max pooling** | Take the maximum across all kernel outputs at each position (most common) |

Max pooling = "did this feature exist anywhere nearby?" — a 1 if any kernel fired strongly at that spot.

---

## NumPy Implementation

The key trick: instead of looping over positions in a Python for-loop at prediction time, **reshape the batch so each patch becomes its own row**. One matrix multiply then covers all kernels at all positions.

```python
def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)

# Collect every 3×3 patch from every position
sects = []
for row_start in range(input_rows - kernel_rows):     # 0..24
    for col_start in range(input_cols - kernel_cols): # 0..24
        sects.append(get_image_section(layer_0,
            row_start, row_start + kernel_rows,
            col_start, col_start + kernel_cols))

# Stack: (batch × 625 positions, 9 pixels each)
expanded_input  = np.concatenate(sects, axis=1)
flattened_input = expanded_input.reshape(batch * 625, -1)

# One dot product = all 16 kernels at all 625 positions for the whole batch
kernel_output = flattened_input.dot(kernels)   # (batch×625, 16)
layer_1 = tanh(kernel_output.reshape(batch, -1))  # (batch, 10 000)
```

The key insight: **a dot product against 16 columns is identical to predicting 16 separate linear layers simultaneously**.

---

## Architecture

```
Input:      (batch, 784)
↓ reshape   (batch, 28, 28)
↓ conv      (batch, 625, 9)  ← 625 patches, 9 pixels each
↓ kernels   (9, 16)          ← 16 kernels with 9 weights each
↓           (batch, 10 000)  ← 625 positions × 16 kernels, flattened
↓ tanh + dropout
↓ weights   (10 000, 10)
↓ softmax   (batch, 10)      ← digit probabilities
```

**Parameter count comparison:**

| Layer type | Parameters |
|---|---|
| Fully connected first layer (784→10 000) | 7,840,000 |
| Convolutional first layer (9×16 kernels) | **144** |

Same expressive power, 54,000× fewer parameters.

---

## Backward Pass Through the Conv Layer

Backprop through a conv layer follows the same rule as always — just reshape carefully:

```python
# layer_1_delta has shape (batch, 10 000)
# kernel_output has shape (batch×625, 16)

l1d_reshape = layer_1_delta.reshape(kernel_output.shape)  # (batch×625, 16)
k_update    = flattened_input.T.dot(l1d_reshape)           # (9, 16)
kernels    -= alpha * k_update
```

The delta flows back through the same flattened representation, then updates the kernel weights.

---

## Results

```
Chapter 9  (tanh + softmax, fully connected):  ~87% test accuracy
Chapter 10 (conv layer + tanh + softmax):       ~87.7% test accuracy
```

The accuracy gain here is modest because we only have 1000 training examples. Convolutions shine at scale — with the full 60,000 MNIST images (or millions of ImageNet images), the gap becomes enormous.

---

## Why This Matters Beyond Images

The structure trick isn't just for images:

| Architecture | Same-idea-in-multiple-places |
|---|---|
| Convolutional (CNN) | Same visual feature detector at every spatial position |
| Recurrent (RNN) | Same word/token processor at every time step |
| Word embeddings | Same meaning representation used wherever a word appears |
| Transformers | Same attention mechanism at every position |

**"When you need the same idea in multiple places — use the same weights."** This principle underlies almost every major deep learning architecture.

---

## Key Vocab

| Term | Plain English |
|---|---|
| **Structure** | Deliberately reusing weights in multiple places to reduce overfitting |
| **Convolutional kernel** | A tiny linear layer (e.g., 3×3) that slides over the entire input |
| **Feature map** | The grid of outputs produced by one kernel scanning an image |
| **Pooling** | Combining multiple kernel outputs into one (sum, mean, or max) |
| **Max pooling** | Taking the highest activation across kernels at each position |
| **Weight sharing** | Using the same kernel weights at every position — the core of convolutions |
| **Receptive field** | The patch of the input a single kernel output depends on (here: 3×3) |

---

## What's Next

Chapter 11 introduces networks that read sequences word-by-word, applying the same idea again — reusing weights at every position in a sequence.

---

[← Chapter 9](../chapter9/README.md) | [Back to Main](../README.md) | [Chapter 11 →](../chapter11/README.md)
