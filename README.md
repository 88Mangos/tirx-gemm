#### Prepping this repo

- [Gemini: Cloning and Pointing upstream to OG repo](https://gemini.google.com/app/80543e3299d68490)
---

# Assignment: Blackwell GEMM Kernel Optimization

In this assignment, you will progressively build a high-performance FP16 GEMM kernel for NVIDIA Blackwell (SM100) GPUs using TVM/TIRX. Starting from a minimal single-tile kernel, you will incrementally add optimizations — K-loop accumulation, spatial tiling, TMA async loads, software pipelining, persistent kernels, warp specialization, deeper pipelines, multi-CTA clusters, and multi-consumer parallelism — until you arrive at a fully optimized kernel that matches the structure of production-grade implementations.

**Prerequisites**: Familiarity with CUDA programming concepts (threads, warps, shared memory, synchronization).

Please read the [slides](https://mlsyscourse.org/slides/tirx-gemm/) for the introduction and guidance to TIRX and this assignment. Reading and fully understanding them beforehand is **strongly recommended**, as they provide important context for the assignment.

---

## Setup

### Modal Setup

1. Install Modal and authenticate with your Andrew email account:

```bash
pip install modal
modal setup
```

2. Run tests via Modal:

```bash
# Run all tests
modal run run_modal.py
# Run a specific step
modal run run_modal.py --step 3
# Run multiple specific steps
modal run run_modal.py --step 1,3,5
```

---

### Local Setup

#### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA Blackwell (B200 / B100) with driver >= 570
- **Python**: >= 3.10 with `pip`

#### Install

```bash
python -m pip install --pre -U -f https://mlc.ai/wheels "mlc-ai-tirx-cu130==0.0.1b2"
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install pytest numpy
```

#### Verify Installation

```bash
python -c "import tvm; print(tvm.__version__)"
python -c "from tvm.script import tirx as Tx; print('TIRX OK')"
```

Both commands should complete without errors.

#### GPU Selection

On multi-GPU machines, select an idle GPU to avoid conflicts with other users:

```bash
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
```

The test framework (`conftest.py`) also auto-selects the least busy GPU if `CUDA_VISIBLE_DEVICES` is not set, but setting it explicitly is recommended.

If tests fail intermittently, check `nvidia-smi` — another process may be using the GPU. Switch to an idle one.

---

## File Structure

```
gemm_kernels.py          # Skeleton — your implementation goes here
utils.py                 # Helpers: prepare_data, compile_and_run, verify, benchmark
run_modal.py             # Run tests on cloud B200 via Modal
inspect_cuda.py          # View generated CUDA/PTX code for any step
tests/
  conftest.py            # Pytest GPU selection fixture
  test_step01.py         # Step 1 test
  ...
  test_step10.py         # Step 10 test
```

---

## How to Work

1. Open `gemm_kernels.py` and implement the `TODO` sections for each step.
2. Run the corresponding test to verify correctness:
   - **Via Modal (cloud B200):**
     ```bash
     modal run run_modal.py --step XX
     # or run multiple steps at once
     modal run run_modal.py --step 1,3,5
     ```
   - **Locally:**
     ```bash
     python -m pytest tests/test_stepXX.py -xvs
     ```
3. Move on to the next step only after the current one passes.
4. Steps are cumulative — each step builds on the previous one. Read the full step description before starting.

---

## Performance Evaluation

GEMM performance is measured in TFLOPS (Tera Floating-Point Operations Per Second):

```
TFLOPS = 2 * M * N * K / (time_in_seconds) / 1e12
```

The factor of 2 accounts for the multiply and add in each fused multiply-add (FMA) operation.

Use the `benchmark` function in `utils.py` to measure your kernel's performance:

```python
from utils import benchmark
from gemm_kernels import hgemm_v10

kernel = hgemm_v10(4096, 4096, 4096)
ms, tflops = benchmark(kernel, 4096, 4096, 4096)
print(f"{ms:.3f} ms, {tflops:.1f} TFLOPS")
```

---

## Grading Rubric

Total: **100 points**. Each step is graded on correctness, performance (within a bound), and implementation.

| Step | Description | Points |
|------|-------------|--------|
| 1 | Single-tile synchronous GEMM | 5 |
| 2 | K-loop accumulation | 5 |
| 3 | Spatial tiling (multi-CTA) | 10 |
| 4 | TMA async load + TMA store | 10 |
| 5 | Software pipeline (PIPE_DEPTH=2) | 10 |
| 6 | Persistent kernel + tile scheduler | 10 |
| 7 | Warp specialization (PIPE_DEPTH=2) | 15 |
| 8 | Deeper pipeline (PIPE_DEPTH=4) | 5 |
| 9 | 2-CTA cluster | 15 |
| 10 | Multi-consumer (final kernel) | 15 |
| **Total** | | **100** |

---

## Submission

Please follow the instructions carefully.

### What is graded

Each step is graded on two criteria:

1. **Correctness** — the kernel output must match cuBLAS reference (within `rtol=1e-3, atol=1e-2`).
2. **Performance** — the kernel must achieve reasonable TFLOP/s compared to the reference implementation. Kernels significantly slower than the reference will fail the performance check.

### Create submission archive

From your assignment root directory, run:

```bash
tar cvf handin.tar gemm_kernels.py
```

You can verify the contents with:

```bash
tar tvf handin.tar
```

It should list exactly one file:

```
-rw-rw-r-- ... gemm_kernels.py
```

### Submission

**Note: Submissions are not open yet. We will provide the submission details later this week.**
