### Step 10: Multi-Consumer Warp Specialization (Final Kernel)
# Setup
```python
# TODO: 3 warpgroups, 2 consumers, 2-CTA cluster.
# Key changes from step 9:
#   - WG_NUMBER=3: WG2 (TMA+MMA), WG0+WG1 (writeback)
#   - NUM_CONSUMER=2 MMA warps (warp0, warp1 in WG2)
#   - Each MMA warp handles tmem[:, warp_id*MMA_N : warp_id*MMA_N+MMA_N]
#   - TMA loads NUM_CONSUMER A blocks per stage
#   - mma2tma.init(NUM_CONSUMER), mma2ld depth=NUM_CONSUMER
#   - WG0/WG1 read from tmem offset by wg_id*MMA_N
#   - Writeback uses per-consumer Dsmem[wg_id, ...]
```
- `WG_NUMBER = 3`, `NUM_CONSUMER = 2`
- Tile scheduler: `num_m_tiles=M // 256 // NUM_CONSUMER` — cluster tile is now 512x256

```python
# --- Hardware Mapping setup  ---
""" 
Given a cluster containing CTA_GROUP=2 CTAs (SMs), which have distributed SMEM,
now the CTAs can also cooperate within the cluster to increase arithmetic intensity.

In both CTA_0 and CTA_1, we consider WG_NUMBER=3 warpgroups.
- WG0 and WG1 are used for writeback
- WG2 is used for TMA loading and MMA 

In each warpgroup, there are 4 warps. 

Let's look at warpgroup 2 first.
- warp 0 and warp 1 are MMA consumers. 
  Warp 0 writes to the first 256 cols of TMEM
  Warp 1 writes to the latter 256 cols of TMEM
  Since TMEM always has 128 lanes, our 256 x 256 result tile is stored as 
  two 128 x 256 tiles side by side, leading to TMEM of 128 x 512,
  i.e., 512 TMEM cols to writeback.
- warp 3 is a TMA producer, which loads 2 A blocks and 1 B block per stage.
  Each block is size 256 x 64.
  Basically, to fill in a 256 x 512 result tile,
  We can do A_tile_0[256 x 64] @ B_tile[64 x 256] in warp 0 -> [256 x 256]
  We can do A_tile_1[256 x 64] @ B_tile[64 x 256] in warp 1 -> [256 x 256]

Now let's think about the writeback WGs.
- WG0 handles the [256 x 256] chunk of TMEM that WG2, warp 0 MMA'd
- WG1 handles the [256 x 256] chunk of TMEM that WG2, warp 1 MMA'd
"""
```
```python
""" 
Hardware Mapping Setup
CTA_GROUP=2-CTA cluster, each CTA gets WG_NUMBER=3 warpgroups.

Tile Scheduling:
  We compute the result in 256x256 tiles, 
    so the number of m_tiles is M // MMA_M=256 // NUM_CONSUMER=2 = M // 512, 
      since each CTA handles a 256x256 output tile, the whole cluster handles 512x256 output tiles.
    and n_tiles is N // MMA_N=256.
  The number of clusters is just the number of SMs used divided by the number of CTAs used per SM.
    Here, we have SM_COUNT=2 SMs and CTA_GROUP=2-CTAs per cluster, so we just have 2 // 2 = 1 cluster.
"""
```
# Shared Memory Allocation and Barrier Setup
```python
""" 
Shared memory allocation

We need to synchronize everything.
  First, let's synchronize TMA and MMA in WG2.
    We have PIPE_DEPTH=4 

    mma2tma should be initialized with expected count NUM_CONSUMER=2, 
      since both MMA consumers should be done before TMA is signaled to start loading the next B block
      (recall that the B block that TMA loads is used by BOTH MMA consumers)
    
  Next, let's synchronize writeback with TMA and MMA.
    mma2ld needs depth=NUM_CONSUMER slots, since both consumers need to signal independently
      to LD that the MMA is done and the result needs to be written back.
    mma2ld should be initialized with expected count 1, since each consumer acts independently
      and writes back its own half of TMEM.

    ld2mma needs NUM_CONSUMER slots too, since the writeback signals to each MMA consumer independently.
    l2dmma should be initialized with expected count 128 * CTA_GROUP=2 = 256, since all 256 threads in
      the cluster need to be done writing back before MMA can recompute a result
      ... is that even true? Why can't I just do the writebacks independently? 
      I guess that's because we re-use the single B block, so write-back is sync'd to one B block.

A, B, and D blocks all have space in SMEM. 
  We load 2 [256 x 64] A blocks since we have 2 consumers, so we need to account for that in ASmem
  We load 1 [64 x 256] B block at each round.
  We writeback D in chunks of size EPI_N, and again, we have 2 consumers, so we have to account for that in DSmem
"""
```
- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` — 2 A blocks per stage
- `mma2tma.init(NUM_CONSUMER)` — each stage expects 2 arrivals (one per MMA warp)
- `mma2ld = TCGen05Bar(pool, NUM_CONSUMER, ...)` and `ld2mma = MBarrier(pool, NUM_CONSUMER, ...)` — one shared object each with `depth=NUM_CONSUMER` (2 slots), **not** separate objects per consumer. Use `warp_id` / `wg_id` as the slot index (not `PipelineState.stage`): `mma2ld.arrive(warp_id, ...)`, `mma2ld.wait(wg_id, ...)`
- `mma2ld.init(1)` — each slot expects 1 arrival (one MMA warp)
- `ld2mma.init(128 * CTA_GROUP)` — each slot expects 256 arrivals (all writeback WG threads across both CTAs)


# TMA Load
- TMA loads both `Asmem[stage, 0, :, :]` and `Asmem[stage, 1, :, :]`

# MMA 
- MMA warp `warp_id` selects which A block: `Asmem[stage, warp_id, :, :]`
- MMA output offset: `tmem[:, warp_id * MMA_N : warp_id * MMA_N + MMA_N]`
- TMA arrive bytes: `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE` — 2 A blocks + 1 B block per CTA

# Writeback 
- Writeback WG offset: `wg_id * MMA_N`

- Writeback **must** use chunked EPI_N (e.g., 64 or smaller) — reading all 256 TMEM columns at once exceeds register capacity

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