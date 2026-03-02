# Chapter 1 — Introducing Deep Learning

## The Core Idea

Deep learning is teaching a machine to find patterns in data — so it can predict, classify, or generate things it has never seen before.

It's not magic. It's math you already know (multiplication, addition, a bit of algebra) applied in a clever, repeated way.

---

## What Is Deep Learning, Actually?

Think of it like this:

> You show a machine 1000 photos of cats and 1000 photos of dogs, each labelled.
> The machine adjusts its internal numbers until it can tell them apart.
> Now show it a photo it's never seen — it guesses correctly.

That process of **adjusting internal numbers based on feedback** is the core loop of every neural network you'll ever build.

"Deep" just means there are multiple layers of these adjustments happening — more on that in Chapter 3.

---

## Why Learn It From Scratch (Not Just Use a Framework)?

Frameworks like TensorFlow and Keras are like driving a car with an automatic gearbox. You can drive, but you don't know what's happening under the hood.

This book (and these notes) teach you the manual gearbox version first — so that when something breaks, or you need to do something unusual, you actually understand what's happening.

**The analogy from the book:** Learning Keras without understanding neural networks is like learning which Chevrolet model is fastest before you know what a gear shift is.

---

## The Learning Approach

Every chapter in Grokking Deep Learning follows this pattern:

1. Memorise a small piece of code from the last chapter
2. Read the new chapter to understand the next concept
3. See it come alive in a working neural network

No heavy maths upfront. Every formula gets an intuitive analogy before the symbols.

---

## What You Need

| Thing | Why |
|---|---|
| Python basics | All examples are in Python |
| NumPy | The only library used for all the maths |
| High school algebra | Multiplication, variables, graphs — that's it |
| A problem you care about | The best motivation for pushing through hard chapters |

---

## Key Vocab to Know Going In

| Term | Plain English |
|---|---|
| **Machine learning** | Teaching a machine using examples rather than rules |
| **Deep learning** | Machine learning using multi-layered neural networks |
| **Neural network** | A system of connected numbers that adjusts itself to reduce error |
| **Framework** | A library (Keras, PyTorch) that handles the deep learning machinery for you |

---

## The Mindset

Chapter 1 is mostly the author telling you: *this is learnable, don't be scared of the math, just build things.*

The one genuinely useful tip: **find a real problem you want to solve.** It doesn't matter what it is. Having a goal makes the hard chapters worth pushing through.

---

## What's Next

Chapter 2 covers the landscape of AI, machine learning, and deep learning — the vocabulary and concepts you need before writing any code. Still no code, but important framing.

Chapter 3 is where you build your first neural network.

---

[← Back to Main](../README.md) | [Chapter 2 →](../chapter2/README.md)
