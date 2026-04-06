### Step 6: Persistent Kernel + Tile Scheduler

**What you will learn:**
- Persistent kernel pattern: fixed number of CTAs that loop over tiles
- `ClusterPersistentScheduler2D` for L2-cache-friendly tile ordering
- Why persistent kernels improve performance

**Background:**

In steps 3-5, each CTA computes exactly one output tile, and the GPU launches `(M/128) * (N/128)` CTAs. For large matrices, this can mean thousands of CTAs, and the launch overhead + cold L2 cache hurt performance.

A persistent kernel launches exactly `SM_COUNT` CTAs (one per SM). Each CTA loops over multiple tiles using a tile scheduler:

```python
tile_scheduler = ClusterPersistentScheduler2D(
    "ts", num_m_tiles=M//128, num_n_tiles=N//128,
    l2_group_size=8, num_clusters=SM_COUNT)
tile_scheduler.init(bx)
while tile_scheduler.valid():
    # ... compute tile at (tile_scheduler.m_idx, tile_scheduler.n_idx) ...
    tile_scheduler.next_tile()
```

The scheduler orders tiles in an L2-cache-friendly pattern (processing nearby tiles together), which significantly improves memory bandwidth utilization.

**Implementation hints:**
- `bx = Tx.cta_id([SM_COUNT], parent="kernel")` — single-dimensional grid.
- `m_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M)`.
- `n_st = Tx.meta_var(tile_scheduler.n_idx * BLK_N)`.
- The K-loop and pipeline logic remain the same as step 5.

**Test:** `pytest tests/test_step06.py -xvs`