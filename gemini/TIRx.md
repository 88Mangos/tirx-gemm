
## TIRX Primer

TIRX is an extended Tensor IR built on top of TVM. It provides a Python DSL for writing GPU kernels that map directly to hardware features. Here is a simplified sketch showing the key elements (not a complete kernel — synchronization and writeback are omitted):

```python
@Tx.prim_func(tirx=True)                    # Declare a TIRX primitive function
def kernel(A: Tx.Buffer((M, K), "float16"),  # Typed buffer parameters
           D: Tx.Buffer((M, N), "float16")):
    with Tx.kernel():                        # Kernel execution scope
        bx, by = Tx.cta_id([grid_m, grid_n], parent="kernel")  # CTA indices
        wg_id = Tx.warpgroup_id([1], parent="cta")             # Warpgroup index
        warp_id = Tx.warp_id([4], parent="warpgroup")          # Warp within WG
        lane_id = Tx.thread_id([32], parent="warp")            # Thread within warp

        pool = Tx.PoolAllocator()            # Shared memory allocator
        Asmem = pool.alloc((128, 64), "float16", layout=A_layout)
        pool.commit()                        # Finalize allocation

        Tx.copy(Asmem[:, :], A[...])         # Synchronous copy GMEM -> SMEM
        Tx.gemm_async(tmem, Asmem, Bsmem,    # Async MMA
                       accum=False, dispatch="tcgen05", cta_group=1)
```

Beyond what the sketch shows, you will need to learn:
- **Scope nesting**: `Tx.kernel()` > `Tx.cta()` > `Tx.warpgroup()` > `Tx.warp()` > `Tx.thread()` control which threads execute a block. For example, `Tx.copy` inside `with Tx.cta():` means all threads cooperate on the copy; inside `with Tx.thread():` means each thread copies independently.
- **`Tx.meta_var`**: Creates compile-time aliases for expressions (e.g., `m_st = Tx.meta_var(bx * 128)`). Use this when you need to pass a computed offset to buffer slicing.
- **`Tx.ptx.*`**: Direct access to PTX intrinsics — the hardware-level operations introduced in the Background section (mbarrier init/arrive/wait, tcgen05 alloc/commit, memory fences).
- **Layouts**: `tma_shared_layout(dtype, SwizzleMode, shape)` creates swizzled layouts for shared memory buffers. You don't need to understand swizzle internals — just pass this layout when allocating SMEM buffers that will be used with TMA or MMA.


### Axe Layout

These kernels use **Axe Layout** ([Hou et al., 2026](https://arxiv.org/abs/2601.19092)), a hardware-aware layout abstraction that maps logical tensor coordinates to named physical axes. Instead of manually computing memory addresses or thread-to-element mappings (as in raw CUDA), you declare a layout on each buffer and the compiler generates the correct address arithmetic, TMA descriptors, and TMEM load instructions automatically.

**Syntax.** The layout spec `S[shape : stride@axis]` reads as "map each dimension to a named hardware axis":

```python
S[(128, 512) : (1@TLane, 1@TCol)]
#  ^^^  ^^^     ^^^^^^^^  ^^^^^^^^
#  rows cols    row axis  col axis
# "128 rows on TLane, 512 cols on TCol"
```

If no `@axis` is given (just a plain number), it defaults to the memory axis `m`.

**Quick reference — layouts used in this kernel:**

| When you need... | Use this | Example buffers |
|---|---|---|
| Shared memory for TMA | `tma_shared_layout(dtype, SWIZZLE_128B_ATOM, shape)` | `Asmem`, `Bsmem`, `Dsmem` |
| TMEM buffer | `TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])` | `tmem` |
| Register view for warpgroup TMEM read | `TileLayout(S[(128, N) : (1@axis_tid_in_wg, 1)])` | `Dreg_wg` |

- **SMEM layout**: `tma_shared_layout` creates a swizzled layout for bank-conflict-free access. You don't need to understand swizzle internals — just call this helper function with your dtype, swizzle mode, and buffer shape.
- **TMEM layout**: `TLane` and `TCol` are Blackwell Tensor Memory's native 2D addressing axes. Declaring this layout tells the compiler the buffer lives in TMEM.
- **Register view**: `axis_tid_in_wg` means "distribute rows across the 128 threads in a warpgroup." When you write `Tx.copy(Dreg_wg, tmem)`, the compiler matches `tid_in_wg` to `TLane` and generates the correct TMEM load instructions.


---
