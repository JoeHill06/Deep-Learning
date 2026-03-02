# Chapter 11 — Word Embeddings: Neural Networks That Understand Language

## The Core Idea

All previous chapters used structured numerical data (pixel values, sensor readings). Language is different — it's variable-length, symbolic, and its meaning is relational. This chapter answers:

1. **How do you turn words into numbers?** (bag of words → embedding layer)
2. **What does the hidden layer learn about language?** (word similarity)
3. **How does the choice of what to predict shape what the network learns?** (loss function = intelligence targeting)

---

## From Text to Numbers: Bag of Words

Neural networks need fixed-size numerical inputs. For text, the simplest encoding is **bag of words**:

- Build a vocabulary of all unique words (e.g., 70,000 words)
- Represent each document as a vector of 0s and 1s — 1 if the word appears, 0 if not
- Word order is discarded; only presence/absence matters

```python
vocab = ['cat', 'the', 'dog', 'sat']

# "the cat sat" → [1, 1, 0, 1]
one_hot_sentence = np.sum([one_hot[w] for w in sentence], axis=0)
```

This is called one-hot encoding. It's the starting point, but it has a problem: for 70,000 words, most of the vector is zeros. Matrix-multiplying a mostly-zero vector is slow.

---

## The Embedding Layer Trick

**One-hot × weight matrix = just selecting and summing rows.**

```python
# Slow: one-hot vector × full matrix
layer_1 = one_hot_vector.dot(weights_0_1)    # mostly multiplying by 0

# Fast: select only the relevant rows and sum them
layer_1 = np.sum(weights_0_1[word_indices], axis=0)
```

These two are mathematically identical. The second is thousands of times faster because it skips all the zero-multiplications.

This is called an **embedding layer**. Instead of a big matrix multiply, you just look up rows by word index.

The weight matrix `weights_0_1` *is* the embedding table — each row is a word's learned vector representation.

---

## IMDB Sentiment: The Full Network

```python
# Forward pass
layer_1 = sigmoid(np.sum(weights_0_1[word_indices], axis=0))  # embed + sum
layer_2 = sigmoid(np.dot(layer_1, weights_1_2))                # predict pos/neg

# Backward pass
layer_2_delta = layer_2 - y
layer_1_delta = layer_2_delta.dot(weights_1_2.T)

# Only update the word rows we actually used — not the whole 70k table
weights_0_1[word_indices] -= layer_1_delta * alpha
weights_1_2               -= np.outer(layer_1, layer_2_delta) * alpha
```

Result: ~85% test accuracy on IMDB after 2 training passes.

---

## What Did the Network Learn?

After training, the embedding rows for similar words will be numerically similar. You can find nearest neighbours using Euclidean distance:

```python
def similar(target):
    target_vec = weights_0_1[word2index[target]]
    scores = {}
    for word, idx in word2index.items():
        diff = weights_0_1[idx] - target_vec
        scores[word] = -math.sqrt(sum(diff * diff))   # negative: higher = closer
    return sorted(scores, key=scores.get, reverse=True)[:10]

similar('terrible') → ['dull', 'boring', 'disappointing', 'annoying', 'horrible', ...]
similar('beautiful') → ['atmosphere', 'heart', 'fascinating', 'beautifully', ...]
```

Words with similar **predictive power** have similar **embeddings**. The network grouped them because doing so made it easier to predict the right label.

---

## Meaning Is Defined by the Loss Function

When trained to predict positive/negative sentiment, "beautiful" and "atmosphere" cluster together — both tend to appear in positive reviews.

When trained to **fill in the blank** (predict a missing word from its context), the clusters change:

```
Predict pos/neg:       similar('beautiful') → atmosphere, heart, tight ...
Fill in the blank:     similar('beautiful') → lovely, creepy, glamorous, cute ...
```

The second clustering is semantically richer — the network needs to understand *what words occur near other words*, which requires a more nuanced model of context.

**The key insight:**

> **The network doesn't learn "data". It minimises a loss function. Change the loss, change what it learns.**

The choice of what to predict, what to measure as error, and what architecture to use — all of these are ways of *designing the loss function*. This is called **intelligence targeting**.

---

## Word Analogies: king − man + woman ≈ queen

A surprising side effect of fill-in-the-blank training: you can do arithmetic on word vectors.

```python
king   = [0.6, 0.1]
man    = [0.5, 0.0]
woman  = [0.0, 0.8]
queen  = [0.1, 1.0]

king - man = [0.1, 0.1]   ← "royalty" direction
queen - woman = [0.1, 0.2] ← same direction
```

Because "king" and "man" co-occur with similar words (except for royalty-related ones), and "queen" and "woman" co-occur with similar words (except for royalty-related ones), the royalty signal ends up in the same dimension of the embedding space.

This is **not magic** — it's a natural consequence of linearly compressing co-occurrence statistics. Any linear model trained on co-occurrences will exhibit this.

---

## The Loss Function — The Central Concept

Everything you can change in a neural network is really a way of shaping the loss function:

| What you change | Effect on loss |
|---|---|
| Training target (pos/neg vs fill-in-blank) | Defines what the error is measuring |
| Architecture (deeper, wider, conv) | Changes how the forward pass transforms input |
| Regularization (dropout, weight decay) | Penalises complexity in the loss |
| Activation functions | Changes what error surface looks like |
| Dataset | Changes what signal is available |

> "If something is going wrong, the solution is in the loss function."

---

## Key Vocab

| Term | Plain English |
|---|---|
| **NLP** | Natural language processing — automated understanding of human language |
| **Bag of words** | Representing a document as a set of unique words (order discarded) |
| **One-hot encoding** | A vector with 1 in one position and 0 everywhere else — one per word |
| **Embedding layer** | A lookup table: map a word index to its learned vector (faster than one-hot × matrix) |
| **Word embedding** | The learned vector representation of a word; its row in the weight matrix |
| **Euclidean distance** | Straight-line distance between two vectors — used to measure word similarity |
| **Negative sampling** | Training against a random subset of the full vocabulary to speed up fill-in-blank training |
| **Loss function** | The full error formula including forward propagation — defines what the network learns |
| **Intelligence targeting** | Choosing the loss function (target, architecture, regularization) to control *what* the network learns |
| **Word analogy** | Arithmetic on word vectors: king − man + woman ≈ queen |

---

## What's Next

Chapter 12 introduces recurrent neural networks (RNNs) — a network that processes sequences one token at a time, feeding its own output back as input. Unlike bag-of-words, RNNs preserve word order.

---

[← Chapter 10](../chapter10/README.md) | [Back to Main](../README.md) | [Chapter 12 →](../chapter12/README.md)
