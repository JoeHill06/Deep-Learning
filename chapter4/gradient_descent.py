# =============================================================================
# CHAPTER 4 — Gradient Descent: How a Network Learns
# This file shows the full evolution from naive learning to gradient descent.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. MEASURING ERROR — Mean Squared Error (MSE)
#    Before the network can learn, it needs to measure how wrong it is.
#    We square the error so:
#      - it's always positive (no cancelling out)
#      - big misses get amplified, tiny misses are suppressed
# -----------------------------------------------------------------------------

weight     = 0.5
input      = 0.5
goal_pred  = 0.8

pred  = input * weight
error = (pred - goal_pred) ** 2

print("=== 1. MSE Error ===")
print(f"Prediction: {pred}  |  Error: {error}\n")


# -----------------------------------------------------------------------------
# 2. HOT AND COLD LEARNING — the naive approach
#    Wiggle the weight up and down a tiny bit.
#    Whichever direction lowers the error — go that way.
#    Problems:
#      - Inefficient: 3 predictions per update
#      - step_amount is arbitrary — can overshoot or never converge
# -----------------------------------------------------------------------------

print("=== 2. Hot and Cold Learning ===")
weight      = 0.5
input       = 0.5
goal_pred   = 0.8
step_amount = 0.001

for iteration in range(5):
    pred  = input * weight
    error = (pred - goal_pred) ** 2

    # Try nudging weight both ways
    up_error   = (goal_pred - input * (weight + step_amount)) ** 2
    down_error = (goal_pred - input * (weight - step_amount)) ** 2

    if down_error < up_error:
        weight -= step_amount
    if up_error < down_error:
        weight += step_amount

    print(f"  iter {iteration}: error={error:.6f}  pred={pred:.4f}  weight={weight:.4f}")

print()


# -----------------------------------------------------------------------------
# 3. GRADIENT DESCENT — the smart approach
#    Compute both direction AND amount in a single formula.
#    No need to try both directions.
#
#    The formula:  weight_delta = (pred - goal_pred) * input
#
#    This does three things automatically:
#      STOPPING         — if input=0, weight_delta=0 (nothing to learn)
#      NEGATIVE REVERSAL — if input<0, sign flips so we still go right direction
#      SCALING          — big input → big update (proportional to sensitivity)
#
#    weight -= alpha * weight_delta
#    alpha (learning rate) controls how big a step we take each iteration
# -----------------------------------------------------------------------------

print("=== 3. Gradient Descent ===")
weight    = 0.0
input     = 0.5
goal_pred = 0.8
alpha     = 0.01    # learning rate — keep this small to avoid overshooting

for iteration in range(20):
    pred         = input * weight
    error        = (pred - goal_pred) ** 2
    delta        = pred - goal_pred          # pure error: how far off, + or -
    weight_delta = delta * input             # derivative: scale & direction fix
    weight      -= alpha * weight_delta      # nudge weight opposite the slope

    if iteration % 5 == 0:
        print(f"  iter {iteration:2d}: error={error:.6f}  pred={pred:.4f}  weight={weight:.4f}")

print()


# -----------------------------------------------------------------------------
# 4. DIVERGENCE — what happens when alpha is too large
#    If input is big, the derivative is steep.
#    A large alpha causes the weight to overshoot every time → explodes.
#    Fix: reduce alpha until learning stabilises.
# -----------------------------------------------------------------------------

print("=== 4. Divergence (alpha too high) ===")
weight    = 0.5
input     = 2.0     # large input → steep derivative
goal_pred = 0.8
alpha     = 1.0     # way too high

for iteration in range(4):
    pred         = input * weight
    error        = (pred - goal_pred) ** 2
    derivative   = input * (pred - goal_pred)
    weight      -= alpha * derivative
    print(f"  iter {iteration}: error={error:.4f}  pred={pred:.4f}")

print()


# -----------------------------------------------------------------------------
# 5. ALPHA FIXES DIVERGENCE
#    Same setup as above, but alpha is now small enough to stay stable.
# -----------------------------------------------------------------------------

print("=== 5. Gradient Descent with Alpha (stable) ===")
weight    = 0.5
input     = 2.0
goal_pred = 0.8
alpha     = 0.1     # try: 0.01, 0.1, 1.0 to see the difference

for iteration in range(20):
    pred       = input * weight
    error      = (pred - goal_pred) ** 2
    derivative = input * (pred - goal_pred)
    weight    -= alpha * derivative

    if iteration % 5 == 0:
        print(f"  iter {iteration:2d}: error={error:.8f}  pred={pred:.6f}")
