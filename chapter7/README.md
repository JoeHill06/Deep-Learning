# Chapter 7 — How to Picture Neural Networks

## The Core Idea

This chapter has no new code. Its entire purpose is to give you a **mental shortcut** — a way to think about neural networks at a higher level so you don't have to hold every detail in your head at once.

The shortcut is called the **correlation summarization**.

---

## The Correlation Summarization

Instead of thinking about individual weights, gradients, and backpropagation steps, compress everything down to one idea:

> **Neural networks find and create correlation between the input layer and the output layer.**

That's it. Everything else — gradient descent, backpropagation, ReLU, alpha — is just machinery that makes this happen.

### Three levels of zoom:

**Global (the whole network):**
> A neural network adjusts its weights to find correlation between the input dataset and the output dataset.

**Layer-to-layer (local):**
> Each weight matrix optimizes to correlate its input layer with what its output layer needs.

**Cross-layer communication (backpropagation):**
> Later layers tell earlier layers what kind of signal they need. Each layer passes the message back: "Send me a higher value." This telephone chain is backpropagation.

Once you trust this summarization, you can **stop thinking about individual neurons and weights** and start thinking about architectures.

---

## The Simplified Visualization

Instead of drawing every node connected by every weight (messy and unreadable at scale), think of a network as **blocks**:

```
[ layer_2  ]   ← vector  (1 × 1)
[ W1       ]   ← matrix  (4 × 1)
[ layer_1  ]   ← vector  (1 × 4)
[ W0       ]   ← matrix  (3 × 4)
[ layer_0  ]   ← vector  (1 × 3)
```

- **Vectors** (layers) = strips of numbers
- **Matrices** (weights) = boxes of numbers

Matrix dimensions are inferred from the layers around them:
- `W0` sits between a layer of size 3 and a layer of size 4 → W0 is (3 × 4)
- `W1` sits between a layer of size 4 and a layer of size 1 → W1 is (4 × 1)

You only need to know the layer sizes. The matrix sizes follow automatically.

---

## Algebraic Notation

You can write an entire forward pass as a single line of algebra:

| Algebra | Meaning |
|---|---|
| `l0W0` | vector-matrix multiply layer_0 by W0 |
| `relu(l0W0)` | apply ReLU to the result |
| `relu(l0W0)W1` | then multiply by W1 to get the output |

The full forward pass of the 3-layer streetlight network:

```
l2 = relu(l0 W0) W1
```

In Python that's:
```python
layer_2 = relu(layer_0.dot(weights_0_1)).dot(weights_1_2)
```

Same thing, three ways — diagram, algebra, code. Pick whichever makes the most sense to you in the moment.

---

## Architecture

The specific arrangement of vectors and matrices in a network is called its **architecture**.

Different architectures suit different problems because they channel signal in different ways:
- Image data needs architectures that detect spatial patterns (convolutional networks)
- Text data needs architectures that handle sequences (recurrent networks)
- Simple tabular data works with fully connected layers

**Good architecture:** channels signal so correlation is easy to discover.
**Great architecture:** also filters noise to prevent overfitting.

The rest of the book is mostly about exploring different architectures and understanding why each one works for its problem type.

---

## Why This Chapter Matters

As networks get larger (hundreds of layers, millions of weights), you cannot think about individual neurons. You'd go insane.

The correlation summarization lets you reason at the right level:
- **Building a new network?** Ask: "How does this architecture help the network find correlation between input and output?"
- **Debugging a failing network?** Ask: "Is the signal reaching the earlier layers? Is there correlation to find at all?"
- **Choosing an architecture?** Ask: "Does this structure match the type of correlation in my data?"

---

## Key Vocab

| Term | Plain English |
|---|---|
| **Correlation summarization** | The high-level mental model: networks find correlation between input and output |
| **Local correlation** | Each weight matrix optimizing for its own layer pair |
| **Global correlation** | Backpropagation communicating what's needed across all layers |
| **Architecture** | The specific configuration of layers and weight matrices |
| **Vector** | A layer — a strip of numbers (lowercase letter in algebra: `l0`, `l1`) |
| **Matrix** | A set of weights — a box of numbers (uppercase letter in algebra: `W0`, `W1`) |

---

## What's Next

Chapters 8 onwards introduce specific architectures designed to handle different kinds of data and correlation. The visualization tools from this chapter will be the common language used to describe all of them.

---

[← Chapter 6](../chapter6/README.md) | [Back to Main](../README.md) | [Chapter 8 →](../chapter8/README.md)
