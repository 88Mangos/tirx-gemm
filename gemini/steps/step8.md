### Step 8: Deeper Pipeline (PIPE_DEPTH=4)

**What you will learn:**
- The effect of pipeline depth on latency hiding
- How to scale the warp-specialized structure to more pipeline stages

**Background:**

Step 7 uses `PIPE_DEPTH=2` (double buffering). With only 2 stages, the TMA producer can be at most 1 stage ahead of the MMA consumer. If the TMA latency is longer than the MMA compute time, the MMA warp stalls waiting for data.

With `PIPE_DEPTH=4`, the TMA producer can be up to 3 stages ahead, providing more buffering to absorb latency variations. The cost is more shared memory (4x the A/B buffers instead of 2x) and more barrier instances.

**Changes from step 7:**
- `PIPE_DEPTH = 4` (was 2)
- `TMABar(pool, 4, ...)`, `TCGen05Bar(pool, 4, ...)`
- `Asmem = pool.alloc((4, BLK_M, BLK_K), ...)`
- `PipelineState("tma", 4)`, `PipelineState("mma", 4)`

Everything else — the warp specialization structure, barrier flow, and epilogue — remains identical.

**Test:** `pytest tests/test_step08.py -xvs`