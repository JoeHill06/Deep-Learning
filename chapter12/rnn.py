"""
Chapter 12 — Recurrent Neural Networks (RNN)

Demonstrates how to:
  1. Show why bag-of-words fails for ordered sequences
  2. Build the identity-matrix stepping trick
  3. Train a linear RNN to predict the next word

Uses a small inline Babi-style dataset so no downloads are needed.
The key idea: a single shared transition matrix ('recurrent') lets
the network build an ordered sentence embedding one word at a time.
"""

import numpy as np
from collections import Counter

np.random.seed(1)


# ─────────────────────────────────────────────
# Part 1: Why bag-of-words loses word order
# ─────────────────────────────────────────────

print("=" * 55)
print("PART 1 — Bag-of-words loses word order")
print("=" * 55)

# Three tiny word vectors (pretend these were learned)
vects = {
    'red':     np.array([1.0, 0.0, 0.5]),
    'sox':     np.array([0.5, 1.0, 0.0]),
    'defeat':  np.array([0.0, 0.5, 1.0]),
    'yankees': np.array([1.0, 0.5, 0.0]),
}

def bag_of_words(words):
    return sum(vects[w] for w in words)

s1 = ['red', 'sox', 'defeat', 'yankees']
s2 = ['yankees', 'defeat', 'red', 'sox']   # opposite meaning, same words

v1 = bag_of_words(s1)
v2 = bag_of_words(s2)

print(f"Sentence 1: {' '.join(s1)}")
print(f"Sentence 2: {' '.join(s2)}")
print(f"Bag-of-words v1: {v1}")
print(f"Bag-of-words v2: {v2}")
print(f"Identical? {np.allclose(v1, v2)}")
print()
print("→ Two sentences with opposite meanings produce the same vector.")
print("  A neural network cannot tell them apart.")


# ─────────────────────────────────────────────
# Part 2: The identity matrix stepping trick
# ─────────────────────────────────────────────

print()
print("=" * 55)
print("PART 2 — Identity stepping (same result, generalisable)")
print("=" * 55)

identity = np.eye(3)

# Summing word vectors directly
direct_sum = vects['red'] + vects['sox'] + vects['defeat']

# Stepping: each word vector is passed through the identity matrix
# before the next word is added — produces the SAME result
step_sum = (((vects['red'].dot(identity) + vects['sox'])
              .dot(identity) + vects['defeat']))

print(f"Direct sum  : {direct_sum}")
print(f"Stepped sum : {step_sum}")
print(f"Identical?    {np.allclose(direct_sum, step_sum)}")
print()
print("→ When the matrix is the identity, order still doesn't matter.")
print("  But if we LEARN a non-identity matrix, order WILL matter.")


# ─────────────────────────────────────────────
# Part 3: Full RNN — predict the next word
# ─────────────────────────────────────────────

print()
print("=" * 55)
print("PART 3 — Recurrent Neural Network on simple sentences")
print("=" * 55)

# Small Babi-style corpus (location statements)
raw_corpus = [
    "mary moved to the kitchen",
    "john went to the hallway",
    "mary went to the garden",
    "daniel moved to the bathroom",
    "john went to the kitchen",
    "sandra went to the garden",
    "mary moved to the hallway",
    "daniel went to the bathroom",
    "john moved to the garden",
    "sandra went to the kitchen",
    "mary went to the bedroom",
    "daniel moved to the hallway",
    "sandra moved to the garden",
    "john went to the bathroom",
    "mary moved to the kitchen",
    "daniel went to the hallway",
    "john moved to the bedroom",
    "mary went to the garden",
    "sandra moved to the kitchen",
    "daniel went to the bedroom",
]

# Tokenise
tokens = [line.split() for line in raw_corpus]
vocab  = list(set(w for s in tokens for w in s))
vocab.sort()
word2index = {w: i for i, w in enumerate(vocab)}

def words2indices(sentence):
    return [word2index[w] for w in sentence]

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

print(f"Vocabulary ({len(vocab)} words): {vocab}")
print()

# ── Hyperparameters ────────────────────────────────────────────────────────
embed_size = 10
alpha      = 0.001

# ── Weights ───────────────────────────────────────────────────────────────
embed     = (np.random.rand(len(vocab), embed_size) - 0.5) * 0.1  # word embeddings
recurrent = np.eye(embed_size)                                     # transition matrix (starts as I)
start     = np.zeros(embed_size)                                   # initial hidden state
decoder   = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1  # hidden → word prediction
one_hot   = np.eye(len(vocab))                                     # for loss computation


# ── Forward propagation ───────────────────────────────────────────────────
def predict(sent):
    """
    Build up a hidden state one word at a time:
      hidden[t] = hidden[t-1] · recurrent + embed[word[t]]
    At each step, predict the NEXT word using the CURRENT hidden state.
    Returns the list of layer dicts and the total cross-entropy loss.
    """
    layers = [{'hidden': start.copy()}]
    loss   = 0.0

    for target_i in range(len(sent)):
        layer = {}
        # Predict next word from current hidden state
        layer['pred']   = softmax(layers[-1]['hidden'].dot(decoder))
        loss           += -np.log(layer['pred'][sent[target_i]] + 1e-12)
        # Advance hidden state: recurrent step + new word embedding
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[sent[target_i]]
        layers.append(layer)

    return layers, loss


# ── Full training loop (forward + backprop + weight update) ──────────────
for iteration in range(30_000):
    sent = words2indices(tokens[iteration % len(tokens)][1:])  # skip sentence index
    layers, loss = predict(sent)

    # ── Backpropagation through time ───────────────────────────────────────
    for layer_idx in reversed(range(len(layers))):
        layer  = layers[layer_idx]
        target = sent[layer_idx - 1]

        if layer_idx > 0:
            layer['output_delta'] = layer['pred'] - one_hot[target]
            new_hidden_delta = layer['output_delta'].dot(decoder.T)

            # Delta at this hidden state = from output + propagated from the future
            if layer_idx == len(layers) - 1:
                layer['hidden_delta'] = new_hidden_delta
            else:
                layer['hidden_delta'] = (new_hidden_delta
                    + layers[layer_idx + 1]['hidden_delta'].dot(recurrent.T))
        else:
            # First layer: no output delta, only incoming from the future
            layer['hidden_delta'] = layers[1]['hidden_delta'].dot(recurrent.T)

    # ── Weight updates ─────────────────────────────────────────────────────
    n = float(len(sent))
    start -= layers[0]['hidden_delta'] * alpha / n

    for layer_idx, layer in enumerate(layers[1:]):
        decoder  -= np.outer(layers[layer_idx]['hidden'],
                             layer['output_delta']) * alpha / n
        embed_idx = sent[layer_idx]
        embed[embed_idx] -= layers[layer_idx]['hidden_delta'] * alpha / n
        recurrent -= np.outer(layers[layer_idx]['hidden'],
                              layer['hidden_delta']) * alpha / n

    # ── Progress ──────────────────────────────────────────────────────────
    if iteration % 5000 == 0:
        print(f"Step {iteration:6d} | Perplexity: {np.exp(loss / len(sent)):.3f}")


# ── Show what the trained network predicts ────────────────────────────────
print()
print("── Sample predictions after training ──────────────────────")
sent_to_test = tokens[2]  # "mary went to the garden"
layers, _ = predict(words2indices(sent_to_test))
print(f"Sentence: {' '.join(sent_to_test)}")
print(f"{'Previous word':<18} {'True next':<18} {'Predicted'}")
print("-" * 54)
for i, layer in enumerate(layers[1:-1]):
    prev_word  = sent_to_test[i]
    true_next  = sent_to_test[i + 1]
    pred_word  = vocab[layer['pred'].argmax()]
    print(f"{prev_word:<18} {true_next:<18} {pred_word}")
