# Lab 2: Digits dataset preparation

The dataset is scikit-learn's built-in `load_digits` dataset. Its 64 pixel
features are scaled to the range 0–1 by dividing by 16, and its integer labels
are one-hot encoded into 10 classes. The data is split into 70% training and
30% test sets, stratified by the original labels, with `random_state=42`.
`load_digits` has no `random_state` parameter; the seed applies to the split.

Run `python src/dataset.py` from the `Lab02` directory to prepare the data and
print the array shapes. The split contains 1,257 training samples and 540 test
samples; features have 64 columns and one-hot labels have 10 columns.

## Network forward pass

`src/network.py` implements the network for hidden widths 16 and 64. For each
width, the weights have shapes `(64, W)`, `(W, W)`, and `(W, 10)` and are
initialized from `N(0, 0.01)`; biases are initialized to zero. The forward
pass applies affine transforms and ReLU to both hidden layers, then computes
a numerically stable softmax over the 10 output classes. Run
`python src/network.py` from `Lab02` to check that an eight-sample batch
produces `(8, 10)` output and each row sums to 1 within `1e-6`.

The same script computes mean cross-entropy from the softmax probabilities
and one-hot labels. Both untrained network widths are checked against
`ln(10)`, the expected loss for near-uniform predictions.

## Backpropagation and gradient check

For mean softmax cross-entropy, the output-logit derivative is
`dL/dz3 = (p - y) / batch_size`. Moving backward through each affine layer
uses `dW = previous_activation.T @ delta` and `db = sum(delta, axis=0)`;
through each ReLU, the incoming derivative is multiplied by the indicator
that its pre-activation is positive. Thus
`delta2 = (delta3 @ W3.T) * 1[z2 > 0]` and
`delta1 = (delta2 @ W2.T) * 1[z1 > 0]`. The implementation returns all three weight and bias gradients.

The script checks 10 randomly selected weight gradients using central
differences with `epsilon=1e-5`. Relative error is
`abs(analytic - numerical) / max(1e-8, abs(analytic) + abs(numerical))`;
at least 9 checks must be below `1e-6` for each width.

## Mini-batch training

The training loop shuffles the training data each epoch and applies gradient
descent in mini-batches of 32 for 50 epochs. It trains all combinations of
hidden widths 16 and 64 and learning rates 0.001, 0.01, and 0.1. Mean
training cross-entropy is logged after every epoch for each run.

The six loss curves are saved to [`results/loss_curves.png`](./results/loss_curves.png).
Final held-out accuracy for each width and learning rate is recorded in
[`results/final_test_accuracy.csv`](./results/final_test_accuracy.csv).

| Hidden width | Learning rate | Final test accuracy |
|---:|---:|---:|
| 16 | 0.001 | 0.1019 |
| 16 | 0.01 | 0.1019 |
| 16 | 0.1 | 0.3259 |
| 64 | 0.001 | 0.1019 |
| 64 | 0.01 | 0.1019 |
| 64 | 0.1 | 0.9389 |

With this initialization and training budget, the 64-unit network at learning
rate 0.1 achieved the highest test accuracy.

## Forward and backward pass timing

Forward inference and backpropagation were timed separately for 100 runs on
the same 32-sample batch. The backward pass reuses activations cached before
timing, so its measurement excludes the forward pass. Times are medians in
milliseconds and are recorded in [`results/pass_timings.csv`](./results/pass_timings.csv).

| Hidden width | Forward median (ms) | Backward median (ms) | Dominant pass |
|---:|---:|---:|---|
| 16 | 0.086200 | 0.095050 | Backward |
| 64 | 0.092700 | 0.116850 | Backward |

Backward took longer for both widths. The forward pass performs three dense
matrix multiplications (one per affine layer). Backpropagation performs
additional matrix multiplications to form each weight gradient and propagate
the error through the hidden layers, along with ReLU derivative and bias
gradient calculations. That extra matrix work explains the measured
backward-pass cost; exact timings depend on batch size and the local NumPy
and BLAS implementation.