## TIRX API Reference

### Thread Hierarchy

| API | Description |
|-----|-------------|
| `Tx.prim_func(tirx=True)` | Declare a TIRX primitive function |
| `Tx.kernel()` | Kernel execution scope |
| `with Tx.cta():` | Scope: all threads in the CTA execute this block |
| `with Tx.warpgroup():` | Scope: all 128 threads in the warpgroup execute this block |
| `with Tx.warp():` | Scope: all 32 threads in the warp execute this block |
| `with Tx.thread():` | Scope: each thread executes independently |
| `with Tx.thread(parent="warp")[cond]:` | Scope: only threads where `cond` is true execute |
| `Tx.cta_id(shape, parent=...)` | CTA index in grid or cluster |
| `Tx.warpgroup_id(shape, parent=...)` | Warpgroup index within CTA |
| `Tx.warp_id(shape, parent=...)` | Warp index within warpgroup |
| `Tx.thread_id(shape, parent=...)` | Thread (lane) index within warp |
| `Tx.ptx.elect_sync()` | Elect one thread in a warp (for single-thread dispatch) |

### Memory

| API | Description |
|-----|-------------|
| `Tx.PoolAllocator()` | Shared memory pool allocator |
| `pool.alloc(shape, dtype, layout=...)` | Allocate buffer from pool |
| `pool.move_base_to(offset)` | Set next allocation offset (for overlapping buffers) |
| `pool.commit()` | Finalize all allocations |
| `Tx.alloc_local(shape, dtype)` | Allocate per-thread register buffer |
| `buf.view(shape, layout=...)` | Create a view of a register buffer with a different layout |
| `Tx.decl_buffer(shape, dtype, scope="tmem", ...)` | Declare a TMEM buffer |
| `Tx.address_of(buf)` | Get address of a buffer (used for TMEM alloc) |
| `buf.ptr_to([idx])` | Get pointer to the idx-th element (used for mbarrier access) |
| `tma_shared_layout(dtype, SwizzleMode, shape)` | Create TMA-compatible swizzled layout for SMEM buffers |
| `Tx.ptx.tcgen05.alloc(addr, n_cols, cta_group)` | Allocate TMEM |
| `Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group)` | Release TMEM allocation permit (call before dealloc) |
| `Tx.ptx.tcgen05.dealloc(addr, n_cols, cta_group)` | Deallocate TMEM |

### Data Movement

| API | Description |
|-----|-------------|
| `Tx.copy(dst, src)` | Synchronous copy |
| `Tx.copy_async(dst, src, dispatch="tma", ...)` | TMA async copy (load or store) |
| `Tx.cast(dst, src)` | Element-wise type cast |
| `Tx.gemm_async(C, A, B, accum, dispatch="tcgen05", cta_group)` | tcgen05 MMA |

### Control Flow

| API | Description |
|-----|-------------|
| `for i in Tx.unroll(N):` | Explicit unrolled loop with `i` usable for buffer slicing |
| `for i in Tx.serial(N):` | Sequential loop (not unrolled), `i` is a TIR variable |
| `Tx.meta_var(expr)` | Compile-time alias for an expression (required for buffer slice offsets) |
| `@Tx.inline` | Decorator for inline helper functions within the kernel |

### Synchronization

| API | Description |
|-----|-------------|
| `Tx.ptx.mbarrier.init(ptr, count)` | Initialize mbarrier with expected arrival count |
| `Tx.ptx.mbarrier.try_wait(ptr, phase)` | Wait for mbarrier phase |
| `Tx.ptx.mbarrier.arrive.expect_tx(ptr, bytes)` | Set expected TMA byte count |
| `Tx.ptx.tcgen05.commit(ptr, cta_group, cta_mask)` | tcgen05 commit (auto-arrive on completion) |
| `Tx.ptx.tcgen05.fence.after_thread_sync()` | Fence before accessing TMEM after sync |
| `Tx.ptx.fence.proxy_async("shared::cta")` | Shared memory fence |
| `Tx.ptx.fence.mbarrier_init()` | Fence after mbarrier initialization |
| `Tx.ptx.cp_async.bulk.commit_group()` | Commit pending TMA store operations |
| `Tx.ptx.cp_async.bulk.wait_group(n)` | Wait until at most `n` TMA store groups remain in flight |
| `Tx.cuda.cta_sync()` | CTA-wide barrier (like `__syncthreads`) |
| `Tx.cuda.warpgroup_sync(barrier_id)` | Warpgroup-level barrier (barrier_id differentiates multiple barriers) |
| `Tx.cuda.cluster_sync()` | Cluster-wide barrier |

### High-Level Abstractions

| API | Description |
|-----|-------------|
| `TMABar(pool, depth, name)` | TMA barrier array (auto-arrive via byte counting) |
| `TCGen05Bar(pool, depth, name)` | tcgen05 barrier array (auto-arrive via commit) |
| `MBarrier(pool, depth, name)` | Manual mbarrier array (threads arrive explicitly) |
| `bar.init(count)` | Initialize barrier with expected arrival count |
| `bar.wait(stage, phase)` | Wait for barrier at given stage and phase |
| `TMABar.arrive(stage, bytes)` | Arrive with expected byte count (TMA load) |
| `TCGen05Bar.arrive(stage, cta_group=, cta_mask=)` | Arrive via tcgen05 commit |
| `MBarrier.arrive(stage, cta_id=, pred=)` | Thread-level arrive |
| `bar.ptr_to([stage])` | Get pointer to barrier at given stage |
| `TMABar.remote_view(cta_id)` | Access another CTA's barrier (for cross-CTA signaling) |
| `PipelineState(name, depth)` | Manages pipeline stage index and phase |
| `PipelineState.init(is_producer=)` | Initialize phase tracking (producer starts ready, consumer starts waiting) |
| `PipelineState.stage` / `.phase` | Current stage index and phase value |
| `PipelineState.move_to_next_stage()` | Advance to next pipeline stage |
| `ClusterPersistentScheduler2D(...)` | L2-friendly tile scheduler for persistent kernels |