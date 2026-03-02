"""
Chapter 11 — Word Embeddings and NLP

Two demos:

1. Embedding lookup trick — shows why selecting rows beats one-hot multiplication
2. IMDB sentiment — trains a network using an embedding layer on movie reviews,
   then shows which words are most similar in embedding space

Requires: keras (for the IMDB dataset)
"""

import numpy as np
import math
from collections import Counter

np.random.seed(1)


# ─────────────────────────────────────────────
# Part 1: The embedding lookup trick
# ─────────────────────────────────────────────

print("=" * 55)
print("PART 1 — Embedding lookup vs one-hot multiplication")
print("=" * 55)

# Tiny vocab: 4 words, each mapped to a 3-dimensional embedding
vocab     = ['cat', 'the', 'dog', 'sat']
word2idx  = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
embed_dim  = 3

# The embedding matrix — each ROW is one word's embedding vector
weights = np.array([
    [0.1, 0.5, 0.2],   # cat
    [0.0, 0.1, 0.0],   # the  (stop word → small values)
    [0.2, 0.4, 0.3],   # dog
    [0.0, 0.2, 0.1],   # sat
])

sentence = ['the', 'cat', 'sat']
indices  = [word2idx[w] for w in sentence]

# Method A: one-hot encode then matrix-multiply (slow for large vocab)
one_hot = np.zeros((len(sentence), vocab_size))
for i, idx in enumerate(indices):
    one_hot[i, idx] = 1
layer_1_slow = one_hot.dot(weights)         # (3, 4) × (4, 3) → (3, 3)
sentence_vec_slow = np.sum(layer_1_slow, axis=0)

# Method B: just select the relevant rows and sum (fast)
sentence_vec_fast = np.sum(weights[indices], axis=0)

print(f"Sentence: {sentence}")
print(f"One-hot method result : {sentence_vec_slow}")
print(f"Row lookup result     : {sentence_vec_fast}")
print(f"Identical: {np.allclose(sentence_vec_slow, sentence_vec_fast)}")
print()
print("For a 70 000-word vocabulary, method B skips ~69 997 "
      "multiplications by zero per word.")


# ─────────────────────────────────────────────
# Part 2: IMDB sentiment with embedding layer
# ─────────────────────────────────────────────

print()
print("=" * 55)
print("PART 2 — IMDB sentiment (embedding + sigmoid)")
print("=" * 55)

from keras.datasets import imdb as keras_imdb

# Keras IMDB: load top-5000 words, reviews are already integer sequences
TOP_WORDS = 5000
(x_train_raw, y_train), (x_test_raw, y_test) = keras_imdb.load_data(
    num_words=TOP_WORDS
)

# Keras prepends a 1 (START token) to every review — strip it
x_train_raw = [r[1:] for r in x_train_raw]
x_test_raw  = [r[1:] for r in x_test_raw]

# Represent each review as the *unique* set of word indices it contains
# (bag-of-words, no duplicates — matches what the book does)
input_train  = [list(set(r)) for r in x_train_raw]
input_test   = [list(set(r)) for r in x_test_raw]
target_train = list(y_train)
target_test  = list(y_test)

print(f"Training reviews : {len(input_train)}")
print(f"Test reviews     : {len(input_test)}")
print(f"Vocab size       : {TOP_WORDS}")
print()

# ── Hyperparameters ───────────────────────────────────────────────────────
alpha       = 0.01
iterations  = 2
hidden_size = 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ── Weights ───────────────────────────────────────────────────────────────
# weights_0_1 IS the embedding matrix: one row per word, one col per hidden unit
weights_0_1 = 0.2 * np.random.random((TOP_WORDS, hidden_size)) - 0.1  # (5000, 100)
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1           # (100, 1)

# ── Training ──────────────────────────────────────────────────────────────
for it in range(iterations):
    correct = total = 0
    for i, (x, y) in enumerate(zip(input_train, target_train)):
        if not x:
            continue

        # Forward — look up and average the word embeddings for this review
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))   # (hidden,)
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))      # (1,)

        # Backward
        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)    # (hidden,)

        # Update only the rows we actually used
        weights_0_1[x] -= layer_1_delta * alpha
        weights_1_2    -= np.outer(layer_1, layer_2_delta) * alpha

        correct += int(abs(layer_2_delta) < 0.5)
        total   += 1

        if i % 2000 == 1999:
            print(f"  Iter {it+1} | {i+1:5d}/{len(input_train)} "
                  f"| Train Acc: {correct/total:.3f}")

    # Test accuracy
    test_correct = test_total = 0
    for x, y in zip(input_test, target_test):
        if not x:
            continue
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
        test_correct += int(abs(layer_2 - y) < 0.5)
        test_total   += 1
    print(f"  Iter {it+1} | Test Acc: {test_correct/test_total:.3f}")


# ─────────────────────────────────────────────
# Part 3: Explore what the embeddings learned
# ─────────────────────────────────────────────

print()
print("=" * 55)
print("PART 3 — Word similarity in embedding space")
print("=" * 55)

# Reverse the Keras index map so we can print actual words
word_index = keras_imdb.get_word_index()
# Keras offsets all indices by 3; index 0/1/2 are reserved tokens
idx2word = {v + 3: k for k, v in word_index.items()}
idx2word[0] = '<PAD>'
idx2word[1] = '<START>'
idx2word[2] = '<UNK>'

def similar(target_word, n=10):
    """Return the n most similar words by Euclidean distance."""
    if target_word not in word_index:
        print(f"  '{target_word}' not in vocabulary")
        return []
    target_idx = word_index[target_word] + 3
    if target_idx >= TOP_WORDS:
        print(f"  '{target_word}' outside top-{TOP_WORDS} words")
        return []
    target_vec = weights_0_1[target_idx]
    scores = Counter()
    for idx in range(TOP_WORDS):
        diff = weights_0_1[idx] - target_vec
        scores[idx] = -math.sqrt(np.dot(diff, diff))
    top = scores.most_common(n + 1)
    return [(idx2word.get(idx, f'<{idx}>'), score) for idx, score in top
            if idx != target_idx][:n]

for query in ['terrible', 'beautiful', 'great']:
    results = similar(query)
    print(f"\nMost similar to '{query}':")
    for word, score in results:
        print(f"  {word:<20s} {score:.4f}")
