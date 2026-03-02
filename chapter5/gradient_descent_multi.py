import numpy as np

# =============================================================================
# CHAPTER 5 — Gradient Descent with Multiple Weights
# Generalising the single-weight update from Ch4 to any network shape.
# Key rule: delta lives on the OUTPUT node, weight_delta = delta * that weight's INPUT
# =============================================================================


# -----------------------------------------------------------------------------
# 1. MULTIPLE INPUTS → SINGLE OUTPUT
#    One delta at the output. Each weight_delta = that weight's input * delta.
#    Weights with bigger inputs get bigger updates (scaling property).
# -----------------------------------------------------------------------------
print("=== 1. Multiple Inputs → Single Output ===")

weights = [0.1, 0.2, -0.1]
alpha   = 0.01

toes  = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
true  = [1, 1, 0, 1]

input = [toes[0], wlrec[0], nfans[0]]
goal  = true[0]

def w_sum(a, b):
    return sum(x * y for x, y in zip(a, b))

def ele_mul(scalar, vector):
    return [scalar * v for v in vector]

for iteration in range(4):
    pred         = w_sum(input, weights)
    error        = (pred - goal) ** 2
    delta        = pred - goal                    # how wrong was the output node?
    weight_deltas = ele_mul(delta, input)         # scale delta by each weight's input

    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]

    print(f"  iter {iteration}: error={error:.6f}  pred={pred:.4f}  weights={[round(w,4) for w in weights]}")

print()


# -----------------------------------------------------------------------------
# 2. SINGLE INPUT → MULTIPLE OUTPUTS
#    One input, multiple output nodes → multiple deltas, one per output.
#    Each weight_delta = shared input * that output's delta.
# -----------------------------------------------------------------------------
print("=== 2. Single Input → Multiple Outputs ===")

weights = [0.3, 0.2, 0.9]   # hurt?, win?, sad?
alpha   = 0.1

wlrec = [0.65, 1.0, 1.0, 0.9]
hurt  = [0.1, 0.0, 0.0, 0.1]
win   = [1.0, 1.0, 0.0, 1.0]
sad   = [0.1, 0.0, 0.1, 0.2]

inp  = wlrec[0]
true = [hurt[0], win[0], sad[0]]

for iteration in range(4):
    pred  = [inp * w for w in weights]
    error = [(p - t) ** 2 for p, t in zip(pred, true)]
    delta = [p - t for p, t in zip(pred, true)]

    # Each output has its own delta; each weight_delta = input * that output's delta
    weight_deltas = [inp * d for d in delta]

    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]

    print(f"  iter {iteration}: errors={[round(e,4) for e in error]}  weights={[round(w,4) for w in weights]}")

print()


# -----------------------------------------------------------------------------
# 3. MULTIPLE INPUTS → MULTIPLE OUTPUTS
#    Need a weight_delta for EVERY weight in the matrix.
#    weight_deltas[i][j] = input[i] * delta[j]
#    This is the OUTER PRODUCT of the input vector and the delta vector.
#
#    Think of it as: every output delta gets scaled by every input independently.
# -----------------------------------------------------------------------------
print("=== 3. Multiple Inputs → Multiple Outputs ===")

#                     toes   win%    fans
weights = np.array([
    [ 0.1,  0.1, -0.3],   # → hurt?
    [ 0.1,  0.2,  0.0],   # → win?
    [ 0.0,  1.3,  0.1],   # → sad?
], dtype=float)

alpha = 0.01

toes  = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
hurt  = [0.1, 0.0, 0.0, 0.1]
win   = [1.0, 1.0, 0.0, 1.0]
sad   = [0.1, 0.0, 0.1, 0.2]

inp  = np.array([toes[0], wlrec[0], nfans[0]])
true = np.array([hurt[0], win[0], sad[0]])

for iteration in range(4):
    pred  = weights.dot(inp)
    error = (pred - true) ** 2
    delta = pred - true

    # Outer product: each input × each delta = full weight_delta matrix
    weight_deltas = np.outer(delta, inp)

    weights -= alpha * weight_deltas

    print(f"  iter {iteration}: errors={np.round(error, 4)}")

print()


# -----------------------------------------------------------------------------
# 4. FROZEN WEIGHT EXPERIMENT
#    What happens if you freeze one weight and let the others learn?
#    Key insight: once error reaches 0, the frozen weight NEVER gets updated —
#    even if it would have been useful. The other weights absorb all the work.
#    This shows how weights can "compensate" for each other.
# -----------------------------------------------------------------------------
print("=== 4. Frozen Weight Experiment (weight[0] frozen) ===")

weights = [0.1, 0.2, -0.1]
alpha   = 0.3

inp  = [toes[0], wlrec[0], nfans[0]]
goal = true[0] if not isinstance(true, np.ndarray) else 1.0
goal = 1.0

for iteration in range(4):
    pred         = w_sum(inp, weights)
    error        = (pred - goal) ** 2
    delta        = pred - goal
    weight_deltas = ele_mul(delta, inp)

    weight_deltas[0] = 0   # freeze weight[0] — it never updates

    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]

    print(f"  iter {iteration}: error={error:.6f}  pred={pred:.4f}  weights={[round(w,4) for w in weights]}")
