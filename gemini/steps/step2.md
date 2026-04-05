### Step 2: K-Loop Accumulation

**What you will learn:**
- Iterating over the K dimension with multiple MMA invocations
- The `accum` flag: `False` for the first K tile, `True` for subsequent tiles (accumulate into existing TMEM values)
- mbarrier phase flipping for repeated synchronization

**Background:**

Real matrices have K >> 64. To handle this, we loop over K in chunks of `BLK_K=64`. Each iteration loads a new (128x64) A tile and (128x64) B tile, then runs an MMA. The `accum` parameter tells the Tensor Core whether to overwrite TMEM (`False`) or add to it (`True`).

The mbarrier is reused across iterations. After each wait, the phase flips (0 -> 1 -> 0 -> ...) so the next wait doesn't immediately succeed on the old arrival.

**Implementation hints:**
- Loop: `for k in range(K_TILES)` where `K_TILES = K // BLK_K`.
- Load: `Tx.copy(Asmem, A[:, k*64:(k+1)*64])`.
- MMA: `accum = (k != 0)` — first iteration is False, rest are True.
- Phase flip: `phase_mma = phase_mma ^ 1` after each wait.

**Test:** `pytest tests/test_step02.py -xvs`
