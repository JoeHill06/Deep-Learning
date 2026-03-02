# Chapter 5 — Generalising Gradient Descent to Multiple Weights

## The Core Idea

Chapter 4 taught gradient descent for a single weight. Chapter 5 shows it works the same way for any number of weights — the rule just repeats:

> **delta lives on the output node. weight_delta = delta × that weight's input.**

No matter how many weights, inputs, or outputs you have, this rule doesn't change.

---

## The Key Terms

| Term | What it is |
|---|---|
| `delta` | How wrong the output node was. `pred - goal`. One per output node. |
| `weight_delta` | How much to move a specific weight. `delta × that weight's input`. One per weight. |

The distinction matters: `delta` belongs to a **node**, `weight_delta` belongs to a **weight**.

---

## Case 1: Multiple Inputs → Single Output

One output node → one `delta`. But you have multiple weights, each with a different input.

```python
pred         = w_sum(input, weights)           # dot product → single number
error        = (pred - goal) ** 2
delta        = pred - goal                     # one delta (output node)
weight_deltas = ele_mul(delta, input)          # delta × each input → one per weight

for i in range(len(weights)):
    weights[i] -= alpha * weight_deltas[i]
```

Each weight gets its own `weight_delta` because each has a different input value. Larger input → larger update (scaling).

---

## Case 2: Single Input → Multiple Outputs

Multiple output nodes → multiple `delta`s (one per output). But only one input, shared by all weights.

```python
pred  = [inp * w for w in weights]             # one prediction per output
delta = [p - t for p, t in zip(pred, true)]   # one delta per output
weight_deltas = [inp * d for d in delta]       # input × each delta

for i in range(len(weights)):
    weights[i] -= alpha * weight_deltas[i]
```

Each weight gets scaled by the same input but a different output delta.

---

## Case 3: Multiple Inputs → Multiple Outputs

Now every weight has its own input AND its own output delta.

The `weight_deltas` form a **matrix** — one entry per weight.
Computing it is the **outer product** of the delta vector and the input vector:

```
weight_deltas[i][j] = delta[i] × input[j]
```

In NumPy:
```python
pred          = weights.dot(inp)               # vector-matrix multiply
delta         = pred - true                    # delta vector (one per output)
weight_deltas = np.outer(delta, inp)           # outer product → delta matrix

weights -= alpha * weight_deltas
```

**What is an outer product?**
Every output delta × every input value = a full grid of weight_deltas. It's like case 1 and case 2 combined.

```
delta  = [d0, d1, d2]        input = [i0, i1, i2]

outer product:
    [ d0*i0,  d0*i1,  d0*i2 ]   ← row 0: all inputs scaled by output 0's delta
    [ d1*i0,  d1*i1,  d1*i2 ]   ← row 1: all inputs scaled by output 1's delta
    [ d2*i0,  d2*i1,  d2*i2 ]   ← row 2: all inputs scaled by output 2's delta
```

---

## The Frozen Weight Experiment

If you freeze one weight and train the others, the other weights compensate and error still reaches 0. But here's the catch:

> **Once error = 0, the frozen weight will never get updated — even after you unfreeze it.**

Why? Because `weight_delta = delta × input`, and `delta = pred - goal`. When error = 0, delta = 0, so weight_delta = 0. Nothing moves.

This reveals a real danger in neural networks: **a weight can be made permanently useless if the rest of the network learns to work around it**. The network found a solution that doesn't need that weight.

---

## Why Do Weights Learn Useful Shapes?

On MNIST (784 pixel inputs → 10 digit outputs), if you visualise the weights for the "2" output as a 28×28 image:
- **Bright pixels** = high positive weight → that pixel strongly predicts "2"
- **Dark pixels** = negative weight → that pixel argues against "2"
- **Neutral** = weight near 0 → that pixel is irrelevant

After training, the weight image looks like a blurry "2". This happens because of the dot product: **the weight vector gets updated to become similar to the input vectors that it should predict positively for**.

A dot product is high when two vectors are similar. So gradient descent pushes the weights to look like the average of the inputs that the output should fire for.

---

## Watch Out: Input Size Affects Learning Speed

A weight with a large input gets larger updates than a weight with a small input (because of scaling). This forces you to use a smaller alpha to prevent the large-input weight from diverging — which slows down learning for all the small-input weights.

**Fix:** Normalise your input data (covered later). Keep all inputs in a similar range (e.g. 0–1) so no single input dominates.

---

## Summary of the Update Rule (works for any shape)

```
1. pred         = forward pass through network
2. error        = (pred - true) ** 2
3. delta        = pred - true               (one per output node)
4. weight_delta = delta × input             (one per weight: outer product if matrix)
5. weight      -= alpha × weight_delta
```

This is gradient descent. It works for 1 weight, 100 weights, or 100,000 weights. The rule is always the same.

---

## What's Next

Chapter 6 introduces **backpropagation** — how to apply this same gradient descent rule to networks with hidden layers (stacked networks). The challenge: how do you compute `delta` for a hidden node when you never directly observe its error?

---

[← Chapter 4](../chapter4/README.md) | [Back to Main](../README.md) | [Chapter 6 →](../chapter6/README.md)
