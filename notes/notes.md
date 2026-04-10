```python
"""
Notes for Tyler (tylery):
To keep the code readable, only comment for
- Synchronization Rules: Explaining why a barrier is placed somewhere.
- Hardware Mapping: Documenting thread/warp/block responsibilities.
- Memory Layouts: High-level notes on tile sizes or swizzling.
Words to comment with:  TODO FIXME XXX NOTE HACK BUG

To avoid writing semantically meaningless code, magic numbers have been declared at the global scope.
NOTE: [from README.md] Constants must be defined outside @Tx.prim_func.
  Variables like EPI_N, TMEM_LD_N, MMA_N must be Python constants defined alongside BLK_M, BLK_K, etc.
  Variables assigned inside the kernel function become TIR dynamic variables, which causes errors when used in buffer slicing.



NOTE: [from README.md, step 4] Use @Tx.inline to define helper functions (e.g., tma_load, mma) inside the kernel.
  These are inlined at compile time and can capture outer variables like Asmem, tma_bar, etc.
  I use "# pyright: ignore[reportUnboundVariable]" whenever Pyright fails to capture the variable and allow it to be modified.

NOTE: A common pattern seen is
  warp_id * THREADS_PER_WARP + lane_id = thread_id
  Recall that warpgroups have 128 threads. So to compute the thread_id,
    warp_id from 0-3 tells us which of the 4 (WARPS_PER_WG) warps the thread belongs to
    mult by 32 (THREADS_PER_WARP) to shift to starting position of the warp given by warp_id
    lane_id adds the threads specific position within its assigned warp.
"""
```