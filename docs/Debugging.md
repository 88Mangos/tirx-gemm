## Common Pitfalls

- **Do NOT use Python `and`/`or` on TIR expressions** (e.g., `warp_id == 0 and lane_id == 0`). These are Python operators that don't work on symbolic TIR variables. Use nested `if` statements instead.
- **`accum` must be boolean-compatible**: Use `False` (not `0`) for the first MMA iteration.
- **Fence API**: Use `Tx.ptx.fence.proxy_async("shared::cta")` — positional argument, not keyword `scope=`.
- **GPU flakiness**: If tests fail intermittently, check `nvidia-smi` and switch to an idle GPU.
- **Dsmem overlap**: `pool.move_base_to(1024)` before Dsmem allows it to overlap with Asmem/Bsmem (reusing memory after MMA is done).
- **Do NOT call `cta_sync()` inside an elected-thread scope** (e.g., inside `Tx.thread()[elect_sync()]`). Only one thread is executing — `cta_sync()` requires all threads to participate, so it will deadlock.
- **`alloc_local` vs `decl_buffer`**: Use `Tx.alloc_local` for register buffers. `Tx.decl_buffer` is only for hardware-managed memory like TMEM. To do cross-thread operations, create a view with `.view()` — but use the original `alloc_local` buffer (not the view) for thread-level operations like `Tx.cast`.
- **TMA store must be followed by `commit_group()` + `wait_group(0)`**: TMA store is asynchronous — without waiting, the next loop iteration may overwrite Dsmem before the store finishes reading it.
- **`fence.after_thread_sync()` required before reading TMEM**: After `mma2ld.wait()` (or any mbarrier wait), you must call `fence.after_thread_sync()` before reading TMEM. Without it, the TMEM data from MMA may not be visible to the reading threads.
- **Constants must be defined outside `@Tx.prim_func`**: Variables like `EPI_N`, `TMEM_LD_N`, `MMA_N` must be Python constants defined alongside `BLK_M`, `BLK_K`, etc. Variables assigned inside the kernel function become TIR dynamic variables, which causes errors when used in buffer slicing.

---

## Debugging: Inspect Generated CUDA Source

Use `inspect_cuda.py` to view the CUDA code the compiler generates from your TIRX kernel. This is the most effective way to debug deadlocks, crashes, and wrong results — it shows you exactly which threads execute which instructions.

### Basic Usage (via Modal)

Use Modal to compile on a cloud B200 — no local GPU required:

```bash
modal run run_modal.py --inspect 7              # Step 7, size 1024 (default)
modal run run_modal.py --inspect 9 --size 2048  # Step 9, size 2048
modal run run_modal.py --inspect 7 > v7.cu      # Save to file

# Search for specific instructions:
modal run run_modal.py --inspect 7 | grep tcgen05_alloc
modal run run_modal.py --inspect 7 | grep mbarrier_init
modal run run_modal.py --inspect 7 | grep -B5 -A5 "tcgen05_alloc"
```

If you have a local Blackwell GPU with the TIRX wheel installed, you can also run directly:

```bash
python inspect_cuda.py 7
python inspect_cuda.py 9 2048
```

### Reading the Generated Code

The generated CUDA kernel starts with a function like:

```c
__global__ void __launch_bounds__(256) kernel_kernel(...) {
  // warp_id_in_cta: 0-7 for 2 warpgroups × 4 warps
  int warp_id_in_cta = __shfl_sync(0xffffffff, (((int)threadIdx.x) >> 5), 0, 32);
  extern __shared__ uchar s_buf_w_offset_ptr[];  // shared memory pool
  ...
```

Key mappings from TIRX to generated CUDA:

| TIRX | Generated CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta % 4) == 0` or `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` internal guard | `((int)threadIdx.x) < 1` (absolute CTA thread 0) |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

---

### Debugging Deadlocks

**Step 1: Confirm it's a deadlock (not a crash)**

```bash
# Deadlock: hangs for ~30s then "unspecified launch failure"
# Crash (XID 43): fails instantly
CUDA_LAUNCH_BLOCKING=1 python -m pytest tests/test_step07.py -xvs -k "1024"
```

**Step 2: Reduce to smallest failing size**

If 1024 passes but 2048+ deadlocks, the bug likely involves pipeline state drift across tiles.

**Step 3: Verify barrier arrival counts match init counts**

| Barrier | `init(count)` | Who arrives | How many |
|---------|---------------|-------------|----------|
| `tma2mma` (TMABar) | `init(1)` | TMA warp via `arrive(stage, bytes)` | 1 |
| `mma2tma` (TCGen05Bar) | `init(1)` | MMA warp via `arrive(stage, cta_group, cta_mask)` | 1 |
| `mma2ld` (TCGen05Bar) | `init(1)` | MMA warp via `arrive(0, cta_group, cta_mask)` | 1 |
| `ld2mma` (MBarrier) | `init(128)` | All WG0 threads via `arrive(0, cta_id, pred)` | 128 |

Common mistakes:
- `ld2mma.init(128)` but `arrive` guarded by `if warp_id == 0: if lane_id == 0:` → only 1 arrival
- Step 10: `mma2ld.init(NUM_CONSUMER)` when each slot only gets 1 arrival → should be `init(1)`

**Step 4: Check barrier inits actually execute**

The `.init()` wrapper uses `threadIdx.x < 1` internally. If you nest it inside `if wg_id == 1:`, no thread satisfies both conditions. Generate code and verify:

```bash
python inspect_cuda.py 7 | grep -B10 "mbarrier_init"
```

Buggy code produces:
```c
if ((warp_id_in_cta >> 2) == 1) {     // wg_id == 1 → threadIdx 128-255
  if ((warp_id_in_cta % 4) == 0) {
    if (((int)threadIdx.x) < 1) {      // threadIdx.x < 1 → only thread 0
      mbarrier_init(...);              // NEVER REACHED — thread 0 is in WG0!
    }
  }
}
```

Correct code produces:
```c
if (((int)threadIdx.x) < 1) {         // at top level, no wg_id guard
  mbarrier_init(...);                  // thread 0 executes this
}
```

**Step 5: Verify TMA and MMA iterate the same number of K tiles**

```bash
python inspect_cuda.py 7 | grep "for (int k"
```

Both loops must show `k < 16` (for K=1024). If MMA shows `k < 15` (`K_TILES - 1`), the barrier phases drift and the second tile deadlocks.

**Step 6: Check `tcgen05.alloc/dealloc` have all 32 lanes participating**

```bash
python inspect_cuda.py 7 | grep -B5 "tcgen05_alloc"
```

Buggy code:
```c
if ((warp_id_in_cta % 4) == 0) {    // warp guard — OK
  if (((int)threadIdx.x) % 32 == 0) { // lane guard — WRONG, only 1 thread
    tcgen05_alloc(...);
  }
}
```

Correct code:
```c
if ((warp_id_in_cta >> 2) == 0) {    // wg guard
  if ((warp_id_in_cta % 4) == 0) {   // warp guard — all 32 lanes execute
    tcgen05_alloc(...);               // no lane guard
  }
}
```

### Debugging Crashes (XID 43 / Illegal Memory Access)

The kernel corrupts the CUDA context, so subsequent CUDA calls (even `torch.randn`) fail.

**Common causes:**

1. **`pool.alloc` after `pool.commit()`** — buffer has invalid SMEM address.
    ```bash
    grep -n "pool\.\|commit\|TMABar\|TCGen05Bar\|MBarrier" gemm_kernels.py
    ```

2. **`tcgen05.alloc/dealloc` with single-thread guard** — see Step 6 above.

3. **Missing `cta_sync()` before `tcgen05.dealloc`** — TMEM freed while writeback still reading. Check the generated code ends with:
    ```c
    tvm_builtin_cuda_cta_sync();           // all threads sync first
    if ((warp_id_in_cta % 4) == 0) {       // then dealloc
      tcgen05_relinquish_alloc_permit();
      tcgen05_dealloc(...);
    }
    ```

### Debugging Wrong Results

Specific mismatch counts (128, 253, 381) indicate synchronization bugs, not arithmetic errors. 128 = one thread's worth of output (one row of a 128-wide tile).

**Common causes:**

1. **Missing `cta_sync()` before `fence.after_thread_sync()`** (Steps 4–6): other threads read TMEM before MMA finishes.
2. **Missing `fence.proxy_async("shared::cta")` before TMA store**: TMA engine doesn't see SMEM writes.
3. **Missing `cp_async.bulk.commit_group()` / `wait_group(0)` after TMA store**: store doesn't complete before SMEM is reused.

---