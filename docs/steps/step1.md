### Step 1: Single-Tile Synchronous GEMM (Warm-up)

**What you will learn:**
- The basic structure of a TIRX kernel: function declaration, thread hierarchy, memory allocation
- Synchronous data loading (GMEM -> SMEM) and tcgen05 MMA invocation
- TMEM allocation/deallocation and writeback (TMEM -> RF -> GMEM)

**Background:**

This is the simplest possible GEMM: the matrix dimensions exactly match one hardware tile (M=128, N=128, K=64), so no tiling or looping is needed. The entire computation is: load A and B into shared memory, run one MMA, read the result from tensor memory to registers, and write to global memory.

The kernel structure is:

1. **Allocate shared memory**: Use `Tx.PoolAllocator()` to allocate `Asmem` (128x64), `Bsmem` (128x64), an mbarrier, and a TMEM address slot.
2. **Allocate TMEM**: `Tx.ptx.tcgen05.alloc(addr, n_cols=512, cta_group=1)` — only warp 0 does this.
3. **Fence + sync**: `fence.proxy_async("shared::cta")` flushes pending shared memory writes, `fence.mbarrier_init()` ensures the mbarrier initialization is visible, and `cta_sync()` synchronizes all threads (like `__syncthreads`). This sequence is needed after initializing barriers and TMEM so that all threads see the results before proceeding.
4. **Load data**: Use `with Tx.cta():` so all 128 threads cooperate on the copy. Then `cta_sync()` + `fence.after_thread_sync()` before MMA (see Synchronization Rules above).
5. **MMA**: Only warp 0's elected thread issues MMA and commit: `if warp_id == 0:` then `with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:`.
6. **Wait for MMA**: `Tx.ptx.mbarrier.try_wait(mma_bar, phase)`. Place this **outside** `if warp_id == 0:` — all threads must wait here, because the subsequent TMEM read requires all 128 threads in the warpgroup to participate after MMA completes.
7. **Writeback**: Two scopes:
   - `with Tx.warpgroup():` — read TMEM to registers (all 128 threads cooperate on TMEM load)
   - `with Tx.thread():` — cast fp32 -> fp16, then write to GMEM. Each of the 128 threads writes one row. A warpgroup has 4 warps of 32 threads, so thread's row is `m_st + warp_id * 32 + lane_id` (warp 0 handles rows 0-31, warp 1 handles rows 32-63, etc.).
8. **Deallocate TMEM**: `tcgen05.relinquish_alloc_permit` + `tcgen05.dealloc`.

**Implementation hints:**
- `accum=False` (not `0`) for the first MMA — TIRX requires a boolean.
- Register buffers: `Tx.alloc_local((BLK_N,), dtype)` allocates a 1D per-thread buffer. Use `.view(128, BLK_N, layout=...)` to create a 2D warpgroup view for TMEM reads.
- **Layouts** (see Axe Layout section above):
  - SMEM: `A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))`
  - TMEM: `TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])`
  - Register view for writeback: `TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)])`

**Test:** `pytest tests/test_step01.py -xvs`