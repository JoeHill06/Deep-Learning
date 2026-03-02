# Chapter 2 — Fundamental Concepts: How Do Machines Learn?

## The Core Idea

Every machine learning algorithm does one thing:

> **It takes an input dataset and transforms it into a more useful output dataset.**

The differences between algorithms come down to two questions:
1. Does it know what the right answer is while learning? (**supervised vs unsupervised**)
2. Does it have a fixed number of adjustable values, or does it grow with data? (**parametric vs nonparametric**)

---

## The Map of AI

```
Artificial Intelligence
└── Machine Learning
    └── Deep Learning  ← this is what we're building
```

Deep learning is not all of AI. It's a specific set of tools (neural networks) inside machine learning, inside the broader field of AI.

---

## Supervised vs Unsupervised

### Supervised Learning — "I'll show you the answer"

You give the machine **input data** and **labelled output data**. It learns the pattern between them.

| Input | Output |
|---|---|
| Pixels of a photo | "cat" or "dog" |
| Monday stock prices | Tuesday stock prices |
| Audio waveform | Text transcript |
| Weather sensor readings | Probability of rain |

The machine adjusts itself until it can predict the output from the input alone.

**This is what we build in this book and what most real-world ML is.**

---

### Unsupervised Learning — "Find the pattern yourself"

You give the machine **one dataset** with no labels. It finds structure (groups/clusters) on its own.

```
Input words:          Output clusters:
puppies  →  1   (cute)
pizza    →  2   (food)
kittens  →  1   (cute)
hot dog  →  2   (food)
burger   →  2   (food)
```

The machine doesn't tell you *what* the clusters mean — just that the groups exist. You figure out the meaning.

**Key insight:** All forms of unsupervised learning are really just clustering in disguise.

---

## Parametric vs Nonparametric

### Parametric — Fixed number of knobs, adjusted by trial and error

Imagine a machine with a set of **knobs**. Each knob controls how sensitive the prediction is to a particular input.

Learning = turning the knobs until predictions get accurate.

```
Steps:
1. PREDICT  — feed data through the current knob settings → get a prediction
2. COMPARE  — measure how wrong the prediction was vs the truth
3. ADJUST   — turn the knobs a little to reduce that error
4. Repeat
```

The number of knobs is **fixed upfront** — you decide how many before training starts.

**Deep learning is parametric.** The "knobs" are called **weights**.

---

### Nonparametric — Counting-based, grows with data

Instead of fixed knobs, these models **count** how often things happen and grow their parameters as they see more data.

Example: a model counting how often each streetlight colour causes cars to go. Three lights = three parameters. Five lights = five parameters. The data decides the size.

| | Parametric | Nonparametric |
|---|---|---|
| Parameters | Fixed upfront | Grows with data |
| Learning style | Trial and error | Counting/probability |
| Examples | Neural networks, linear regression | Decision trees, KNN |
| Used in this book? | Yes | No |

---

## The 3-Step Learning Loop (Supervised Parametric)

This is the fundamental loop behind every neural network you'll build:

```
1. PREDICT   → pass input through current weights → get a guess
2. COMPARE   → how far off was the guess from the truth?
3. LEARN     → nudge the weights in the direction that reduces the error
```

Repeat thousands of times. That's training.

---

## Where Deep Learning Fits

Deep learning is:
- **Supervised** (most of the time) — it learns from labelled examples
- **Parametric** — it has a fixed set of weights (knobs) adjusted by trial and error
- **Neural network based** — the weights are organised in connected layers

---

## Key Vocab

| Term | Plain English |
|---|---|
| **Supervised learning** | Learning with labelled answers provided |
| **Unsupervised learning** | Learning by finding hidden structure, no labels |
| **Parametric** | Fixed number of adjustable values (weights) |
| **Nonparametric** | Number of values grows based on the data |
| **Parameters / Weights** | The numbers inside a model that get adjusted during training |
| **Clustering** | Grouping data points by similarity — the output of unsupervised learning |

---

## What's Next

Chapter 3 is where the book stops talking and starts building. You'll write your first neural network from scratch — a single neuron that makes a prediction, measures its error, and updates its weight. All three steps of the loop above, live in code.

---

[← Chapter 1](../chapter1/README.md) | [Back to Main](../README.md) | [Chapter 3 →](../chapter3/README.md)
