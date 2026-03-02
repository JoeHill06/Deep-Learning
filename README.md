# Deep Learning from Scratch

Neural network implementations built from scratch in NumPy, following along with **[Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning)** by Andrew W. Trask (Manning Publications).

The goal is to understand the mechanics of deep learning by implementing everything by hand — no high-level frameworks for the actual learning logic.

---

## Examples

### `trafficLights.py` — Street Light Classifier

A minimal two-layer network that learns to predict walk/stop from streetlight patterns.

**Features used:**
- Forward propagation through two weight layers
- Mean squared error loss
- Backpropagation with hand-computed deltas
- ReLU activation (`relu`, `relu2deriv`)
- Stochastic gradient descent (one sample at a time)

**Architecture:** `3 inputs → 4 hidden → 1 output`

---

### `mnist.py` — Handwritten Digit Classifier

A two-layer network trained on the MNIST dataset (1000 training images) that classifies digits 0–9.

**Features used:**
- Mini-batch gradient descent (`batch_size = 100`)
- Dropout regularization (50% drop rate, rescaled by 2 to preserve expected values)
- ReLU activation
- One-hot encoded labels
- Train/test split evaluation printed every 10 iterations

**Architecture:** `784 inputs → 100 hidden → 10 output`

**Performance:** ~85% train accuracy, ~80% test accuracy at iteration 290

---

## Chapter Notes

Each chapter folder contains a README that teaches the concepts from scratch, plus working Python examples.

| Chapter | Topic | Code |
|---|---|---|
| [Chapter 1](chapter1/README.md) | Introducing Deep Learning | — |
| [Chapter 2](chapter2/README.md) | How Do Machines Learn? | — |
| [Chapter 3](chapter3/README.md) | Forward Propagation | [`forward_propagation.py`](chapter3/forward_propagation.py) |
| [Chapter 4](chapter4/README.md) | Gradient Descent | [`gradient_descent.py`](chapter4/gradient_descent.py) |
| [Chapter 5](chapter5/README.md) | Multiple Weights | [`gradient_descent_multi.py`](chapter5/gradient_descent_multi.py) |
| [Chapter 6](chapter6/README.md) | Backpropagation | [`backpropagation.py`](chapter6/backpropagation.py) |
| [Chapter 7](chapter7/README.md) | Visualising Networks | — |
| [Chapter 8](chapter8/README.md) | Regularization & Batching | [`regularization.py`](chapter8/regularization.py) |
| [Chapter 9](chapter9/README.md) | Activation Functions | [`activation_functions.py`](chapter9/activation_functions.py) |
| [Chapter 10](chapter10/README.md) | Convolutional Neural Networks | [`cnn.py`](chapter10/cnn.py) |
| [Chapter 11](chapter11/README.md) | Word Embeddings & NLP | [`word_embeddings.py`](chapter11/word_embeddings.py) |
| [Chapter 12](chapter12/README.md) | Recurrent Neural Networks | [`rnn.py`](chapter12/rnn.py) |

---

## Concepts Covered (from Grokking Deep Learning)

| Concept | Where used |
|---|---|
| Gradient descent | Both examples |
| Backpropagation | Both examples |
| ReLU & its derivative | Both examples |
| Mini-batch training | `mnist.py` |
| Dropout regularization | `mnist.py` |
| One-hot encoding | `mnist.py` |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
python trafficLights.py
python mnist.py
```
