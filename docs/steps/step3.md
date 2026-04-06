### Step 3: Spatial Tiling (Multi-CTA)

**What you will learn:**
- Launching a 2D grid of CTAs to cover arbitrary M and N dimensions
- Per-CTA tile offset calculation

**Background:**

Steps 1-2 only handle M=N=128. To support larger matrices, we launch a 2D grid of CTAs: `[M // BLK_M, N // BLK_N]`. Each CTA computes one 128x128 output tile.

CTA `(bx, by)` computes `D[bx*128 : (bx+1)*128, by*128 : (by+1)*128]` by loading `A[bx*128 : (bx+1)*128, :]` and `B[by*128 : (by+1)*128, :]`.

**Implementation hints:**
- `bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")`
- `m_st = Tx.meta_var(bx * BLK_M)`, `n_st = Tx.meta_var(by * BLK_N)`
- The K-loop body is the same as step 2, just with offset A and B slices.

**Test:** `pytest tests/test_step03.py -xvs`