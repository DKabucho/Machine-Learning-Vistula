# Lab 05: Scalar Automatic Differentiation and Execution Performance

## 1. Scalar reverse-mode engine

[`src/autodiff.py`](./src/autodiff.py) implements a scalar `Value` node with
its numeric data, accumulated gradient, parent nodes, operation label, and
local backward rule. Addition, multiplication, scalar powers, ReLU, sigmoid,
logarithm, division, and scalar sum build graph nodes during the forward pass.
`backward()` constructs a topological ordering, seeds the output derivative
with 1, and visits nodes in reverse order. Each local rule adds its contribution
to parent gradients, including when a parent is used more than once.

The implementation also supports repeated calls to `backward()`: each call
computes a fresh set of local derivatives and adds them to the gradients already
stored on the graph. The tests cover shared-parent accumulation, power and ReLU
derivatives, and repeated backward calls.

## 2. Gradient validation

[`src/run_lab.py`](./src/run_lab.py) evaluates
`((a*b + c)**2).relu().sum()` at `(a, b, c) = (0.7, -1.2, 0.4)`. It compares
the custom engine against PyTorch autograd and central finite differences
(step size `1e-6`):

| Input | Custom engine | PyTorch | Numerical |
|---|---:|---:|---:|
| `a` | 1.0560000000 | 1.0560000000 | 1.0560000000 |
| `b` | -0.6160000000 | -0.6160000000 | -0.6160000000 |
| `c` | -0.8800000000 | -0.8800000000 | -0.8800000000 |

Maximum absolute difference is **0** versus PyTorch and **7.45e-11** versus
the numerical estimate; both satisfy the required `< 1e-6` tolerance.

## 3. `make_moons` training benchmark

Both implementations train a `2 -> 8 -> 1` model with ReLU hidden activation,
sigmoid output, binary cross entropy, and SGD (learning rate `0.08`) for 10
full-batch epochs. The generated `make_moons` dataset has 192 samples; a
stratified 75/25 split provides 144 training examples. Each implementation is
warmed up once, then timed in five runs. The following are the five-run
medians on the CPU recorded in
[`results/lab05_results.json`](./results/lab05_results.json):

| Implementation | Median training time | Median final loss |
|---|---:|---:|
| Custom scalar autodiff | 1.3005 s | 0.6475 |
| PyTorch | 0.0083 s | 0.5791 |

PyTorch was about **157.4x faster** in this small experiment. This is an
implementation-specific timing, not a general performance guarantee: the
custom engine creates and traverses a Python graph for individual scalars,
while PyTorch executes vectorized tensor operations. The two runs use
deterministic but separately initialized parameters, so final losses are not
intended as a direct quality comparison.

## 4. PyTorch graph, tracing, and tensor layout

The script inspects `Linear(2, 8) -> ReLU -> Linear(8, 1) -> sigmoid` with
`torch.fx.symbolic_trace`. The graph is recorded in the results JSON. The
model was also traced using `torch.jit.trace`; after 20 warm-up calls, 200
forward passes per sample, and five timed samples, the median times were:

| Mode | Median forward time | Shape / dtype |
|---|---:|---|
| Eager | 71.2 us | `(64, 2)` / `float32` |
| `torch.jit.trace` | 61.4 us | `(64, 2)` / `float32` |

The eager and traced outputs matched exactly in this run. Tracing is used
rather than `torch.compile`; the traced execution path was available in the
installed PyTorch build.

For layout, the same elementwise operation (`relu(tensor * 1.01)`) was timed
on a `(512, 512)` transpose view and a contiguous copy. The view is
`float32`, has stride `(1, 512)`, and is non-contiguous; the copied tensor has
stride `(512, 1)`. With five samples of 100 operations each, the medians were
**285.5 us** (non-contiguous) and **279.1 us** (contiguous), so the view was
about **2.3% slower** in this CPU run. The measured difference is small and
hardware- and operation-dependent.

## 5. Reproducing and validating

Run the unit tests from the repository root:

```powershell
python -m pytest Lab05\tests -q
```

Run the checks and regenerate benchmark results from the `Lab05` directory:

```powershell
python -m src.run_lab
```

The environment, gradient checks, graph, per-run timings, tensor metadata, and
summary ratios are saved to
[`results/lab05_results.json`](./results/lab05_results.json). Timings vary
between machines and runs.