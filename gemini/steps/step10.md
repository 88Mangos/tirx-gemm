### Step 10: Multi-Consumer Warp Specialization (Final Kernel)

**What you will learn:**
- Multiple MMA warps (consumers) for higher throughput
- Multiple writeback warpgroups
- How the reference production kernel is structured

**Background:**

The final optimization adds a second MMA consumer. With `NUM_CONSUMER=2` and `WG_NUMBER=3`:

- **WG2**: Producer warpgroup
  - warp 0: MMA consumer 0 — computes `A[0:128, :] x B` -> TMEM columns `[0:256]`
  - warp 1: MMA consumer 1 — computes `A[128:256, :] x B` -> TMEM columns `[256:512]`
  - warp 3: TMA producer — loads 2x A blocks + 1x B block per stage
- **WG0**: Writeback for consumer 0 (reads TMEM `[0:256]`)
- **WG1**: Writeback for consumer 1 (reads TMEM `[256:512]`)

This doubles the compute density per CTA: each CTA now processes a 256x256 output tile (vs 128x256 in step 9), and the cluster output becomes 512x256 (vs 256x256 in step 9).

**Changes from step 9:**
- `WG_NUMBER = 3`, `NUM_CONSUMER = 2`
- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` — 2 A blocks per stage
- TMA loads both `Asmem[stage, 0, :, :]` and `Asmem[stage, 1, :, :]`
- MMA warp `warp_id` selects which A block: `Asmem[stage, warp_id, :, :]`
- MMA output offset: `tmem[:, warp_id * MMA_N : warp_id * MMA_N + MMA_N]`
- Writeback WG offset: `wg_id * MMA_N`
- `mma2tma.init(NUM_CONSUMER)` — each stage expects 2 arrivals (one per MMA warp)
- `mma2ld = TCGen05Bar(pool, NUM_CONSUMER, ...)` and `ld2mma = MBarrier(pool, NUM_CONSUMER, ...)` — one shared object each with `depth=NUM_CONSUMER` (2 slots), **not** separate objects per consumer. Use `warp_id` / `wg_id` as the slot index (not `PipelineState.stage`): `mma2ld.arrive(warp_id, ...)`, `mma2ld.wait(wg_id, ...)`
- `mma2ld.init(1)` — each slot expects 1 arrival (one MMA warp)
- `ld2mma.init(128 * CTA_GROUP)` — each slot expects 256 arrivals (all writeback WG threads across both CTAs)
- Writeback **must** use chunked EPI_N (e.g., 64 or smaller) — reading all 256 TMEM columns at once exceeds register capacity
- Tile scheduler: `num_m_tiles=M // 256 // NUM_CONSUMER` — cluster tile is now 512x256
- TMA arrive bytes: `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE` — 2 A blocks + 1 B block per CTA

**Test:** `pytest tests/test_step10.py -xvs`