# Chapter 12 — Recurrent Neural Networks: Learning From Sequences

## The Core Idea

Bag-of-words treats every word independently and throws away order. "Red Sox defeat Yankees" and "Yankees defeat Red Sox" produce **identical** vectors — the network cannot tell them apart.

An RNN fixes this by processing words **one at a time**, carrying a running "memory" (the hidden state) that is updated at every step. Because the update uses a matrix multiplication, changing the order of words changes the hidden state — and therefore changes the prediction.

---

## Why Bag-of-Words Fails for Sequences

```
"Red Sox defeat Yankees" → avg([red] + [sox] + [defeat] + [yankees])
"Yankees defeat Red Sox" → avg([yankees] + [defeat] + [red] + [sox])
                         = same vector (addition is associative)
```

Any two sentences that use the same words (regardless of order) map to the same vector. That means a neural network looking at those vectors **cannot** distinguish them.

---

## Step 1: The Identity Matrix Trick

Before jumping to RNNs, consider this rewrite of the sum:

```python
# Direct sum
sentence_vec = word_a + word_b + word_c

# Stepped sum (identical result because matrix = identity)
sentence_vec = ((word_a.dot(identity) + word_b).dot(identity) + word_c)
```

These two are mathematically the same. But the second version has a **hook**: what if the matrix wasn't the identity?

---

## Step 2: Make the Matrix Learnable — That's an RNN

Replace the fixed identity matrix with a **learned** matrix called `recurrent`. Now the hidden state evolves as:

```
hidden[0] = start                                  # empty sentence
hidden[t] = hidden[t-1] · recurrent + embed[word[t]]
```

- `hidden[t]` is the network's representation of everything it has read so far
- `recurrent` is shared across **all** timesteps (the structure trick again — same weights in multiple places)
- `embed[word]` is the word's embedding vector (looked up by index)

Because `recurrent` is not the identity, the order in which words are fed in **changes** the resulting hidden state. The network learns which orderings matter.

---

## Architecture

```
start (zeros)
  ↓
hidden[0] · recurrent + embed[word[0]] → hidden[1]
                                              ↓
hidden[1] · recurrent + embed[word[1]] → hidden[2]
                                              ↓
                                           ...
hidden[t] · decoder → softmax → predict next word
```

Three sets of learnable weights:

| Weight | Shape | Role |
|---|---|---|
| `embed` | (vocab, embed_size) | Word embeddings |
| `recurrent` | (embed_size, embed_size) | Transition — starts as identity |
| `decoder` | (embed_size, vocab) | Hidden state → next-word prediction |

---

## Forward Propagation in Python

```python
def predict(sent):
    layers = [{'hidden': start.copy()}]
    loss   = 0.0

    for target_i in range(len(sent)):
        layer = {}
        # Predict the next word from the CURRENT hidden state
        layer['pred']   = softmax(layers[-1]['hidden'].dot(decoder))
        loss           += -np.log(layer['pred'][sent[target_i]])

        # Advance the hidden state: apply recurrent then add new word
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[sent[target_i]]
        layers.append(layer)

    return layers, loss
```

At each step, the network does two things simultaneously:
1. **Predicts** the next word using the hidden state built from all previous words
2. **Updates** the hidden state by mixing the current hidden state with the new word

---

## Backpropagation Through Time (BPTT)

The network is "unrolled" in time. Each timestep is just another layer. The delta flows backwards through the sequence:

```python
# At the last timestep: delta comes from the prediction only
layer['hidden_delta'] = output_delta.dot(decoder.T)

# At earlier timesteps: delta from prediction + delta from the future
layer['hidden_delta'] = (output_delta.dot(decoder.T)
                         + next_layer['hidden_delta'].dot(recurrent.T))
```

When you add two things together in the forward pass (the recurrent step and the word embedding), the gradient flows back through **both branches**.

---

## Weight Updates

```python
# Decoder: hidden state → word prediction
decoder -= np.outer(hidden[t], output_delta[t]) * alpha

# Recurrent: accumulates from every timestep
recurrent -= np.outer(hidden[t-1], hidden_delta[t]) * alpha

# Word embedding: only the word used at this step
embed[word_idx] -= hidden_delta[t] * alpha
```

`recurrent` collects gradients from every step in the sequence — this is the weight-reuse idea from Chapter 10, now applied in time.

---

## Perplexity

The book tracks **perplexity** instead of accuracy:

```
perplexity = exp(cross_entropy_loss / sentence_length)
```

| Value | Meaning |
|---|---|
| Very high (e.g. 80) | Network is guessing randomly |
| Close to 1 | Network is nearly certain of the correct word |

Perplexity decreases as training progresses — same intuition as any loss function going down.

---

## What the RNN Learns

After training, a sentence like "mary moved to the ___" gets progressively better predictions:

```
Early training:   "the" → "the" (just picks the most common word)
Later training:   "moved" → "to", "to" → "the", "the" → "bedroom"
```

The network learns that certain verbs ("moved", "went") are followed by "to the", and that "to the" is followed by a location. It picks up these bigram and trigram patterns from the data.

---

## Key Limitation: No Nonlinearity

This RNN has no activation function — the hidden state is a purely linear combination of previous states and the new word. This means:

- It can only capture linear patterns
- It tends to forget words from earlier in the sequence
- Long-range dependencies are lost

Chapter 13 fixes this with **nonlinearities** and **gates** → the LSTM.

---

## Key Vocab

| Term | Plain English |
|---|---|
| **Recurrent** | Applied repeatedly — the same weights used at every timestep |
| **Hidden state** | The network's running memory of the sequence so far |
| **BPTT** | Backpropagation Through Time — unrolled backprop over the sequence |
| **Perplexity** | e^(loss/length) — how surprised the model is by the actual next word |
| **Transition matrix** | The `recurrent` weight matrix that evolves the hidden state |
| **Unrolling** | Visualising an RNN as a chain of layers (one per timestep) for backprop |

---

## What's Next

Chapter 13 adds **nonlinearities and gates** to the RNN. Instead of just multiplying and adding, it learns which information to **remember**, which to **forget**, and which to **output** at each step. This is the Long Short-Term Memory (LSTM) network.

---

[← Chapter 11](../chapter11/README.md) | [Back to Main](../README.md)
