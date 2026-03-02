# Chapter 4 — Gradient Descent: How a Network Learns

## The Core Idea

A network learns by doing one thing repeatedly:

> **Measure how wrong the prediction was. Nudge the weight in the direction that reduces that error.**

This chapter is the "compare" and "learn" steps of the Predict → Compare → Learn loop.

---

## Step 1: Compare — Measuring Error

Before the network can fix itself, it needs to know how wrong it was.

```python
error = (pred - goal_pred) ** 2
```

This is called **Mean Squared Error (MSE)**. We square the difference for two reasons:

1. **Always positive** — `pred - goal_pred` can be negative. If errors cancelled each other out across many examples, you'd think the network was perfect when it wasn't.
2. **Amplifies big mistakes** — A miss of 10 becomes an error of 100. A miss of 0.01 becomes 0.0001. This makes the network prioritise fixing large errors first.

---

## Step 2: Learn — Three Ways to Update the Weight

### Method 1: Hot and Cold (Naive)

Try wiggling the weight up a tiny bit. Try wiggling it down a tiny bit. Whichever gives lower error — go that way.

```python
up_error   = (goal - input * (weight + step)) ** 2
down_error = (goal - input * (weight - step)) ** 2

if down_error < up_error:
    weight -= step
if up_error < down_error:
    weight += step
```

**Problems:**
- 3 predictions per update — inefficient
- Fixed step size can overshoot or never converge

---

### Method 2: Gradient Descent (The Real Way)

Calculate direction AND amount in one formula. No guessing.

```python
delta        = pred - goal_pred       # pure error: positive = predicted too high
weight_delta = delta * input          # the derivative
weight      -= alpha * weight_delta   # step opposite the slope
```

**Why multiply by `input`?** It does 3 things at once:

| Property | What it does |
|---|---|
| **Stopping** | If `input == 0`, update = 0. Nothing to learn when input is silent. |
| **Negative reversal** | If `input < 0`, the sign flips — still moves weight the right way. |
| **Scaling** | Big input → bigger update. Proportional to how sensitive the prediction is. |

---

## The Full Learning Loop

```python
pred         = input * weight          # 1. PREDICT
error        = (pred - goal) ** 2      # 2. COMPARE (measure the miss)
delta        = pred - goal             # 3. pure error
weight_delta = delta * input           # 4. derivative (direction + amount)
weight      -= alpha * weight_delta    # 5. LEARN (nudge the weight)
```

Repeat this thousands of times. Error approaches 0. That's training.

---

## What Is a Derivative?

A derivative answers: **"When I change this variable, how much does that variable change?"**

Imagine two rods sticking out of a box. Push the blue rod 1 inch — the red rod moves 2 inches. The derivative of red with respect to blue is 2.

In neural networks:
- The derivative of `error` with respect to `weight` tells you the slope of the error curve at the current weight
- **Positive slope** → weight is too high, move it down
- **Negative slope** → weight is too low, move it up
- **Move opposite the slope** to reach the minimum error

This is called **gradient descent** — you descend the error curve by following the slope downhill.

---

## Alpha — The Learning Rate

If the derivative (slope) is very steep, a single update can overshoot the minimum and bounce to the other side — getting worse with each step. This is called **divergence**.

```
alpha too high  →  divergence (error explodes)
alpha too low   →  learns but very slowly
```

Fix: multiply the update by a small number `alpha` before applying it.

```python
weight -= alpha * weight_delta
```

**Finding alpha in practice:** try orders of magnitude — `0.1`, `0.01`, `0.001` — and watch what happens to the error. If it goes up, alpha is too high. If it barely moves, alpha is too low.

---

## The Error Bowl

The relationship between `weight` and `error` always forms a U-shaped curve (for MSE):

```
error
  |       *               *
  |          *         *
  |             *   *
  |               * ← error = 0 (perfect weight)
  |_________________________ weight
```

- The slope at any point tells you which direction to move
- The steeper the slope, the farther you are from the minimum
- Gradient descent walks you down this bowl step by step

---

## Key Vocab

| Term | Plain English |
|---|---|
| **Mean Squared Error (MSE)** | `(pred - goal)²` — always positive, amplifies big mistakes |
| **Pure error / delta** | `pred - goal` — the raw signed miss |
| **Derivative / weight_delta** | How much the error changes when you nudge the weight |
| **Gradient descent** | Moving weight opposite the slope to find minimum error |
| **Alpha / learning rate** | A small multiplier that controls step size to prevent overshooting |
| **Divergence** | When updates overshoot and error spirals upward instead of down |

---

## What's Next

Chapter 4 only trains on a single input-output pair. Chapter 5 extends this to learning from real datasets with multiple examples, adding the ability to **generalise** — not just memorise one answer.

---

[← Chapter 3](../chapter3/README.md) | [Back to Main](../README.md) | [Chapter 5 →](../chapter5/README.md)
