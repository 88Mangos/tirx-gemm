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


Glossary:
  Acronyms:
    GMEM/HBM    : High-Bandwidth Global Memory
    SMEM        : Shared Memory (per-thread)
    SM          : streaming multiprocessor
    TMA         : tensor memory accelerator
    CTA         : collaborative thread array
    WG          : warpgroup
    MMA         : matrix multiply and accumulate
    GEMM        : general matrix multiplication

  Common Variable Patterns:
    tmem_addr   : Slot to store the TMEM base address returned by tcgen05.alloc
    mma_bar     : mbarrier for MMA completion signaling
    tma_bar     : mbarrier for TMA completion signaling
    *_st        : [*] stride

NOTE: [from README.md, step 4] Use @Tx.inline to define helper functions (e.g., tma_load, mma) inside the kernel.
  These are inlined at compile time and can capture outer variables like Asmem, tma_bar, etc.
  I use "# pyright: ignore[reportUnboundVariable]" whenever Pyright fails to capture the variable and allow it to be modified.

NOTE: A common pattern seen is
  warp_id * THREADS_PER_WARP + lane_id = thread_id
  Recall that warpgroups have 128 threads. So to compute the thread_id,
    warp_id from 0-3 tells us which of the 4 (WARPS_PER_WG) warps the thread belongs to
    mult by 32 (THREADS_PER_WARP) to shift to starting position of the warp given by warp_id
    lane_id adds the threads specific position within its assigned warp.


TODO: refactor all the mbarriers to use the semantically correct ones instead of manual allocation.
Will probably have to tinker with the pyi?

TODO: refactor inlined functions to make semantic sense: wait -> command -> arrival queued

TODO: make gemm_async use MMA_N, not BLK_N
"""

import tvm
from tvm.script import tirx as Tx

from tvm.tirx.op_schedule.cuda.common import tma_shared_layout, SwizzleMode
from tvm.tir.layout import TileLayout, S, TLane, TCol, tid_in_wg as axis_tid_in_wg
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.pipeline import PipelineState, MBarrier, TMABar, TCGen05Bar


# MARK: Constants
# ======================================================================
# GEMM Constants
# All TIRx functions here solve A [M x K] @ B [K x N] -> D [M x N]
# ======================================================================
a_type = tvm.DataType("float16")
b_type = tvm.DataType("float16")
d_type = tvm.DataType("float16")
acc_type = tvm.DataType("float32")

BLK_M, BLK_N, BLK_K = 128, 128, 64

# ======================================================================
# Hardware Constants
# ======================================================================
F16_SIZE = 2

# B200-specific
SM_COUNT = 148
TMEM_LANES = 128
WG_PER_CTA = 1
WARPS_PER_WG = 4
THREADS_PER_WARP = 32
THREADS_PER_WG = THREADS_PER_WARP * WARPS_PER_WG


# ======================================================================
# Step 1: Single-tile synchronous GEMM
#   M=128, N=128, K=64 — exactly one tile, no loops.
#   All threads sync-load GMEM→SMEM, one MMA, sync writeback.
# ======================================================================
# MARK: Step 1
def hgemm_v1(M, N, K):
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            _wg_id = Tx.warpgroup_id([WG_PER_CTA], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)  # Skip to offset 1024 so data buffers don't overlap with barriers

            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                # Allocate 512 TMEM columns. address_of() passes the address where the HW writes the TMEM base.
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            # Flush shared memory writes, ensure mbarrier init is visible, then sync all threads
            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            # Declare a logical view of the allocated TMEM (allocated_addr=0 means use the base from tcgen05.alloc)
            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            m_st = Tx.meta_var(bx * BLK_M)  # Compile-time alias for tile row offset
            n_st = Tx.meta_var(by * BLK_N)  # Compile-time alias for tile col offset

            phase_mma: Tx.int32  # NOTE: TIR requires explicit type declaration for mutable variables
            phase_mma = 0

            # --- Synchronous Load ---
            with Tx.cta():
                Tx.copy(Asmem[:, :], A[m_st : m_st + BLK_M, 0:BLK_K])
                Tx.copy(Bsmem[:, :], B[n_st : n_st + BLK_N, 0:BLK_K])

            # SYNC RULE: Threads write SMEM -> MMA reads SMEM
            Tx.cuda.cta_sync()
            Tx.ptx.tcgen05.fence.after_thread_sync()

            # --- Single-thread dispatch MMA Instr ---
            # (warp 0 only, single elected thread), then commit and wait on mma_bar
            if warp_id == 0:
                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :], accum=False, dispatch="tcgen05", cta_group=1)
                    Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # all threads wait until MMA complete -> queued tcgen05.commits -> phase flip
            Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

            # SYNC RULE: MMA writes TMEM -> Threads read TMEM
            Tx.cuda.cta_sync()
            Tx.ptx.tcgen05.fence.after_thread_sync()

            # --- Writeback: TMEM → RF → GMEM ---

            # Allocate local (per-thread) register space for the f32 TMEM results and f16 output.
            Dreg = Tx.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = Tx.alloc_local((BLK_N,), d_type)

            # Apply Axe Layout: map 128 threads_per_warpgroup (axis_tid_in_wg) to 128 rows per tile
            Dreg_wg = Dreg.view(BLK_M, BLK_N, layout=TileLayout(S[(THREADS_PER_WG, BLK_N) : (1 @ axis_tid_in_wg, 1)]))

            with Tx.warpgroup():  # TMEM read (tcgen05.ld) requires all 128 threads to cooperate
                Tx.copy(Dreg_wg[:, :], tmem[:, :BLK_N])  # TMEM → registers
                Tx.cuda.cta_sync()

            with Tx.thread():  # math/GMEM writes executed independently per-thread
                Tx.cast(Dreg_f16[:], Dreg[:])
                row = m_st + warp_id * THREADS_PER_WARP + lane_id  # Calculate the exact global matrix row this specific thread is responsible for.
                Tx.copy(D[row, n_st : n_st + BLK_N], Dreg_f16[:])

            # --- TMEM cleanup ---
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 2: K-loop — accumulate in TMEM
#   M=128, N=128, K=any multiple of 64.
#   Loop over K dimension with accumulation.
# ======================================================================
# MARK: Step 2
def hgemm_v2(M, N, K):
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            _wg_id = Tx.warpgroup_id([WG_PER_CTA], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)

            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            m_st, n_st = Tx.meta_var(bx * BLK_M), Tx.meta_var(by * BLK_N)

            phase_mma: Tx.int32
            phase_mma = 0

            for k in range(K_TILES):
                # --- Sync-load A[:, k*BLK_K : (k+1)*BLK_K] and B[:, ...] to SMEM ---
                with Tx.cta():
                    Tx.copy(Asmem[:, :], A[:, k * BLK_K : (k + 1) * BLK_K])
                    Tx.copy(Bsmem[:, :], B[:, k * BLK_K : (k + 1) * BLK_K])

                # SYNC RULE: Threads write SMEM -> MMA reads SMEM
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

                # --- Issue MMA with accum = (k != 0) ---
                if warp_id == 0:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :], accum=(k != 0), dispatch="tcgen05", cta_group=1)
                        Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

                # --- Wait on mma_bar, flip phase ---
                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

                # SYNC RULE: MMA writes TMEM -> Threads read TMEM
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()
                phase_mma = phase_mma ^ 1  # flip phase

            # --- Writeback TMEM → RF → GMEM (same as step 1) ---
            Dreg = Tx.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = Tx.alloc_local((BLK_N,), d_type)

            Dreg_wg = Dreg.view(BLK_M, BLK_N, layout=TileLayout(S[(THREADS_PER_WG, BLK_N) : (1 @ axis_tid_in_wg, 1)]))

            with Tx.warpgroup():
                Tx.copy(Dreg_wg[:, :], tmem[:, :BLK_N])
                Tx.cuda.cta_sync()

            with Tx.thread():
                Tx.cast(Dreg_f16[:], Dreg[:])
                row = m_st + warp_id * THREADS_PER_WARP + lane_id
                Tx.copy(D[row, n_st : n_st + BLK_N], Dreg_f16[:])

            # --- TMEM cleanup ---
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 3: Spatial tiling — multi-CTA
#   M, N any multiples of 128, K any multiple of 64.
#   Grid of (M/128)×(N/128) CTAs.
# ======================================================================
# MARK: Step 3
def hgemm_v3(M, N, K):

    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            # Launch (M/BLK_M) × (N/BLK_N) CTAs
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            _wg_id = Tx.warpgroup_id([WG_PER_CTA], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)

            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            # Use bx*BLK_M and by*BLK_N as tile offsets.
            m_st, n_st = Tx.meta_var(bx * BLK_M), Tx.meta_var(by * BLK_N)

            phase_mma: Tx.int32
            phase_mma = 0

            # The rest is like step 2 but with dynamic m_st, n_st.
            for k in range(K_TILES):
                # --- Sync-load A[:, k*BLK_K : (k+1)*BLK_K] and B[:, ...] to SMEM ---
                with Tx.cta():
                    # NOTE: different from step 2, since m_st and n_st are dynamic now!
                    Tx.copy(Asmem[:, :], A[m_st : m_st + BLK_M, k * BLK_K : (k + 1) * BLK_K])
                    Tx.copy(Bsmem[:, :], B[n_st : n_st + BLK_N, k * BLK_K : (k + 1) * BLK_K])

                # SYNC RULE: Threads write SMEM -> MMA reads SMEM
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

                # --- Issue MMA with accum = (k != 0) ---
                if warp_id == 0:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :], accum=(k != 0), dispatch="tcgen05", cta_group=1)
                        Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

                # --- Wait on mma_bar, flip phase ---
                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

                # SYNC RULE: MMA writes TMEM -> Threads read TMEM
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()
                phase_mma = phase_mma ^ 1  # flip phase

            # --- Writeback TMEM → RF → GMEM (same as step 1) ---
            Dreg = Tx.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = Tx.alloc_local((BLK_N,), d_type)

            Dreg_wg = Dreg.view(BLK_M, BLK_N, layout=TileLayout(S[(THREADS_PER_WG, BLK_N) : (1 @ axis_tid_in_wg, 1)]))

            with Tx.warpgroup():
                Tx.copy(Dreg_wg[:, :], tmem[:, :BLK_N])
                Tx.cuda.cta_sync()

            with Tx.thread():
                Tx.cast(Dreg_f16[:], Dreg[:])
                row = m_st + warp_id * THREADS_PER_WARP + lane_id
                Tx.copy(D[row, n_st : n_st + BLK_N], Dreg_f16[:])

            # --- TMEM cleanup ---
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 4: TMA async load
#   Replace sync load with TMA (single-thread dispatch, mbarrier sync).
#   Writeback uses TMA store: TMEM → RF → SMEM → TMA → GMEM.
# ======================================================================
# MARK: Step 4
def hgemm_v4(M, N, K):

    K_TILES = K // BLK_K

    """ 
    MMA_N     : output tile width 
    EPI_N     : epilogue width. This is a TMA Store Constraint. 
                Why 64? Blackwell's TMA engine and SMEM layouts (like SWIZZLE_128B_ATOM) often perform best when writing back data in specific power-of-two widths.
                In your code, you process the 128-wide tile in two 64-column "slices" to fit into the Dsmem buffer you allocated. 
                It prevents you from needing a massive 128x128 SMEM buffer for the output, saving shared memory.
    TMEM_LD   : Register Load Width. This is the Thread-Level Granularity.
                When moving data TMEM → Registers, the warpgroup (128 threads) reads a small "strip" of columns.
                8 is chosen because each thread can easily hold 8 float32 values in its local registers (Dreg = Tx.alloc_local((8,), "float32")).
                If you increased this to 16, each thread would need more registers, potentially hitting "Register Pressure" limits which slows down the GPU.
    """
    MMA_N = BLK_N

    # NOTE: taken from hgemm_v7 starter code comments
    EPI_N = 64  # Optional, can be any value that divides MMA_N (e.g., 64, 128)
    TMEM_LD_N = 8  # Optional, can be any value that divides MMA_N (e.g., 8, 16, 128)

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            _wg_id = Tx.warpgroup_id([WG_PER_CTA], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")
            thread_id = Tx.meta_var(warp_id * THREADS_PER_WARP + lane_id)

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma_bar = pool.alloc((1,), "uint64", align=8)
            mma_bar = pool.alloc((1,), "uint64", align=8)

            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)

            # NOTE: Reset the base to 1024 so Dsmem reuses Asmem/Bsmem space.
            # Exists because I tried setting EPI_N to 128 originally,
            # BLK_M * BLK_K + BLK_N + BLK_K + BLK_M * EPI_N elements
            # = 128 * 64 * 2 + 128 * 128 = 128 * 128 * 2 = 32,768 elements,
            # F16_SIZE = 2 bytes per element, exceeds 48KB SMEM limit.
            # Re-use means we only use 128 * 128 * 2 = 32768 bytes, under the 48KB limit.
            pool.move_base_to(1024)
            Dsmem = pool.alloc((BLK_M, EPI_N), d_type, layout=D_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                    Tx.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            m_st, n_st = Tx.meta_var(bx * BLK_M), Tx.meta_var(by * BLK_N)

            phase_tma: Tx.int32
            phase_mma: Tx.int32
            phase_tma = 0
            phase_mma = 0

            # ── TMA async load: GMEM → SMEM (hardware-driven) ──
            @Tx.inline
            def tma_load(k_st):
                byte_count = (BLK_M * BLK_K + BLK_N * BLK_K) * 2

                Tx.copy_async(Asmem[:, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma_bar.ptr_to([0]))
                Tx.copy_async(Bsmem[:, :], B[n_st : n_st + BLK_N, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma_bar.ptr_to([0]))
                Tx.ptx.mbarrier.arrive.expect_tx(tma_bar.ptr_to([0]), byte_count)

            # ── MMA compute: wait for data, run tcgen05, wait for result ──
            @Tx.inline
            def mma(accum):
                # ---  Waits on tma_bar (data ready) ---
                Tx.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)  # pyright: ignore[reportUnboundVariable]  # noqa: F823
                phase_tma ^= 1  # pyright: ignore[reportUnboundVariable]  # noqa: F841
                Tx.ptx.tcgen05.fence.after_thread_sync()

                # --- Issues gemm_async + commit ---
                Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :], accum=accum, dispatch="tcgen05", cta_group=1)
                Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)  # signal MMA done

                # --- Waits on mma_bar (MMA done) ---
                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)  # pyright: ignore[reportUnboundVariable] # noqa: F823
                phase_mma ^= 1  # pyright: ignore[reportUnboundVariable]  # noqa: F841

            # --- Main loop (elected thread of warp 0) ---
            with Tx.thread(parent="warpgroup")[thread_id == 0]:
                for k in range(K_TILES):
                    tma_load(k * BLK_K)
                    mma(k != 0)

            # SYNC RULE: MMA writes TMEM -> Threads read TMEM
            Tx.cuda.warpgroup_sync(10)
            Tx.ptx.tcgen05.fence.after_thread_sync()

            # ── Writeback: TMEM → Reg → SMEM → GMEM ──
            Dreg_f16 = Tx.alloc_local((MMA_N,), d_type)

            for no in Tx.unroll(MMA_N // TMEM_LD_N):
                no_st = Tx.meta_var(no * TMEM_LD_N)
                Dreg = Tx.alloc_local((TMEM_LD_N,), acc_type)  # tmp register

                # --- all threads in WG write TMEM → Reg ---
                with Tx.warpgroup():
                    Dreg_wg = Dreg.view(MMA_N, TMEM_LD_N, layout=TileLayout(S[(MMA_N, TMEM_LD_N) : (1 @ TLane, 1 @ TCol)]))
                    Tx.copy(Dreg_wg[:, :], tmem[:, no_st : no_st + TMEM_LD_N])

                # -- per-thread cast ---
                with Tx.thread():
                    Tx.cast(Dreg_f16[no_st : no_st + TMEM_LD_N], Dreg[:])

            for no in Tx.unroll(MMA_N // EPI_N):
                no_st = Tx.meta_var(no * EPI_N)
                n_epi_st = Tx.meta_var(n_st + no_st)

                # --- per-thread write EPI_N-col slice to SMEM ---
                with Tx.thread():
                    Tx.copy(Dsmem[thread_id, :], Dreg_f16[no_st : no_st + EPI_N])

                    # make SMEM writes visible to TMA engine
                    Tx.ptx.fence.proxy_async("shared::cta")

                # wait until all threads done writing SMEM
                Tx.cuda.warpgroup_sync(10)

                # --- TMA Store (elected thread of warp 0) ---
                with Tx.thread(parent="warpgroup")[thread_id == 0]:
                    Tx.copy_async(D[m_st : m_st + BLK_M, n_epi_st : n_epi_st + EPI_N], Dsmem[:, :], dispatch="tma")

                    # Commit and wait for TMA store completion
                    Tx.ptx.cp_async.bulk.commit_group()
                    Tx.ptx.cp_async.bulk.wait_group(0)

                # Sync before next iteration
                Tx.cuda.warpgroup_sync(10)

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 5: Software pipeline
#   PIPE_DEPTH=2 multi-buffered SMEM. Prefetch + overlap.
# ======================================================================
# MARK: Step 5
def hgemm_v5(M, N, K):

    K_TILES = K // BLK_K
    PIPE_DEPTH = 2

    MMA_N = BLK_N

    # NOTE: cannot exceed PIPE_DEPTH.
    # Buffer Overwrite: If you issue a 3rd TMA load when you only have 2 SMEM buffers, you will overwrite data that the Tensor Core is likely still reading for the 1st tile.
    # Mbarrier Ambiguity: mbarrier uses a 1-bit phase (0 or 1) to distinguish between "new data arrived" and "old data consumed."
    #   If you have 2 buffers, you can track them easily.
    #   If you tried to prefetch 3 tiles into 2 buffers, the hardware wouldn't know which "arrival" the barrier is signaling.
    PRE_NUM = min(PIPE_DEPTH, K_TILES)

    # NOTE: taken from hgemm_v7 starter code comments
    EPI_N = 64  # Optional, can be any value that divides MMA_N (e.g., 64, 128)
    TMEM_LD_N = 8  # Optional, can be any value that divides MMA_N (e.g., 8, 16, 128)

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    # Setup thread hierarchy, allocate PIPE_DEPTH-buffered SMEM,
    # init PIPE_DEPTH mbarriers for TMA, 1 for MMA.
    #
    # Pipeline pattern:
    #   1. Prefetch PRE_NUM stages
    #   2. Main loop: mma(stage) then tma_load(next_stage)
    #   3. Track phase_tma[stage] per stage, phase_mma globally
    #
    # Writeback same as step 4.
    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            _wg_id = Tx.warpgroup_id([WG_PER_CTA], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")
            thread_id = Tx.meta_var(warp_id * THREADS_PER_WARP + lane_id)

            # --- Shared memory allocation ---

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
            mma_bar = pool.alloc((1,), "uint64", align=8)

            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)

            # NOTE: Reset the base to 1024 so Dsmem reuses Asmem/Bsmem space.
            pool.move_base_to(1024)
            Dsmem = pool.alloc((BLK_M, EPI_N), d_type, layout=D_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                    for i in range(PIPE_DEPTH):
                        Tx.ptx.mbarrier.init(tma_bar.ptr_to([i]), 1)

                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            m_st, n_st = Tx.meta_var(bx * BLK_M), Tx.meta_var(by * BLK_N)

            phase_tma: Tx.int32
            phase_mma: Tx.int32
            phase_tma = 0
            phase_mma = 0

            # ── TMA async load: GMEM → SMEM (hardware-driven) ──
            @Tx.inline
            def tma_load(k_st, stage):
                byte_count = (BLK_M * BLK_K + BLK_N * BLK_K) * 2

                # NOTE: MBar now needs to point to the current stage
                Tx.copy_async(Asmem[stage, :, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma_bar.ptr_to([stage]))
                Tx.copy_async(Bsmem[stage, :, :], B[n_st : n_st + BLK_N, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma_bar.ptr_to([stage]))
                Tx.ptx.mbarrier.arrive.expect_tx(tma_bar.ptr_to([stage]), byte_count)

            # ── MMA compute: wait for data, run tcgen05, wait for result ──
            @Tx.inline
            def mma(accum, stage):
                # ---  Waits on tma_bar (data ready) ---
                Tx.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)  # pyright: ignore[reportUnboundVariable]  # noqa: F823
                if stage == PIPE_DEPTH - 1:  # phase flips when stage would wrap to zero
                    phase_tma ^= 1  # pyright: ignore[reportUnboundVariable]  # noqa: F841
                Tx.ptx.tcgen05.fence.after_thread_sync()

                # --- Issues gemm_async + commit ---
                # NOTE: for the current stage
                Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :], accum=accum, dispatch="tcgen05", cta_group=1)
                Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)  # signal MMA done

                # --- Waits on mma_bar (MMA done) ---
                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)  # pyright: ignore[reportUnboundVariable] # noqa: F823
                phase_mma ^= 1  # pyright: ignore[reportUnboundVariable]  # noqa: F841

            # --- Main loop (elected thread of warp 0) ---
            with Tx.thread(parent="warpgroup")[thread_id == 0]:
                # Prefetch PRE_NUM stages into SMEM
                for pre_k in range(PRE_NUM):
                    stage = pre_k % PIPE_DEPTH
                    pre_k_st = pre_k * BLK_K

                    tma_load(pre_k_st, stage)

                # For each K tile, wait for load to finish, compute, then issue the next load.
                for k in range(K_TILES):
                    stage = k % PIPE_DEPTH

                    # Waits on load to finish, then compute the current tile
                    mma(k != 0, stage)

                    # Issue the load for the future tile into the newly freed stage
                    next_k = k + PRE_NUM
                    if next_k < K_TILES:
                        tma_load(next_k * BLK_K, stage)

            # SYNC RULE: MMA writes TMEM -> Threads read TMEM
            Tx.cuda.warpgroup_sync(10)
            Tx.ptx.tcgen05.fence.after_thread_sync()

            # ── Writeback: TMEM → Reg → SMEM → GMEM ──
            Dreg_f16 = Tx.alloc_local((MMA_N,), d_type)

            for no in Tx.unroll(MMA_N // TMEM_LD_N):
                no_st = Tx.meta_var(no * TMEM_LD_N)
                Dreg = Tx.alloc_local((TMEM_LD_N,), acc_type)  # tmp register

                # --- all threads in WG write TMEM → Reg ---
                with Tx.warpgroup():
                    Dreg_wg = Dreg.view(MMA_N, TMEM_LD_N, layout=TileLayout(S[(MMA_N, TMEM_LD_N) : (1 @ TLane, 1 @ TCol)]))
                    Tx.copy(Dreg_wg[:, :], tmem[:, no_st : no_st + TMEM_LD_N])

                # -- per-thread cast ---
                with Tx.thread():
                    Tx.cast(Dreg_f16[no_st : no_st + TMEM_LD_N], Dreg[:])

            for no in Tx.unroll(MMA_N // EPI_N):
                no_st = Tx.meta_var(no * EPI_N)
                n_epi_st = Tx.meta_var(n_st + no_st)

                # --- per-thread write EPI_N-col slice to SMEM ---
                with Tx.thread():
                    Tx.copy(Dsmem[thread_id, :], Dreg_f16[no_st : no_st + EPI_N])

                    # make SMEM writes visible to TMA engine
                    Tx.ptx.fence.proxy_async("shared::cta")

                # wait until all threads done writing SMEM
                Tx.cuda.warpgroup_sync(10)

                # --- TMA Store (elected thread of warp 0) ---
                with Tx.thread(parent="warpgroup")[thread_id == 0]:
                    Tx.copy_async(D[m_st : m_st + BLK_M, n_epi_st : n_epi_st + EPI_N], Dsmem[:, :], dispatch="tma")

                    # Commit and wait for TMA store completion
                    Tx.ptx.cp_async.bulk.commit_group()
                    Tx.ptx.cp_async.bulk.wait_group(0)

                # Sync before next iteration
                Tx.cuda.warpgroup_sync(10)

            Tx.cuda.cta_sync()

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 6: Persistent kernel + tile scheduler
#   Fixed SM_COUNT CTAs, loop over tiles with L2-friendly ordering.
# ======================================================================
# MARK: Step 6
def hgemm_v6(M, N, K):

    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    PRE_NUM = min(PIPE_DEPTH, K_TILES)

    MMA_N = BLK_N

    # NOTE: taken from hgemm_v7 starter code comments
    EPI_N = 64  # Optional, can be any value that divides MMA_N (e.g., 64, 128)
    TMEM_LD_N = 8  # Optional, can be any value that divides MMA_N (e.g., 8, 16, 128)

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    # Launch SM_COUNT persistent CTAs.
    # Use ClusterPersistentScheduler2D for tile iteration.
    #
    # Key changes from step 5:
    #   - bx = Tx.cta_id([SM_COUNT], parent="kernel")
    #   - tile_scheduler = ClusterPersistentScheduler2D(...)
    #   - while tile_scheduler.valid(): ... tile_scheduler.next_tile()
    #   - m_st/n_st from tile_scheduler.m_idx/n_idx

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            tile_scheduler = ClusterPersistentScheduler2D("ts", num_m_tiles=M // 128, num_n_tiles=N // 128, l2_group_size=8, num_clusters=SM_COUNT)
            tile_scheduler.init(bx)

            _wg_id = Tx.warpgroup_id([WG_PER_CTA], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")
            thread_id = Tx.meta_var(warp_id * THREADS_PER_WARP + lane_id)

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
            mma_bar = pool.alloc((1,), "uint64", align=8)

            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)

            # NOTE: Reset the base to 1024 so Dsmem reuses Asmem/Bsmem space.
            pool.move_base_to(1024)
            Dsmem = pool.alloc((BLK_M, EPI_N), d_type, layout=D_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                    for i in range(PIPE_DEPTH):
                        Tx.ptx.mbarrier.init(tma_bar.ptr_to([i]), 1)

                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            phase_tma: Tx.int32
            phase_mma: Tx.int32
            phase_tma = 0
            phase_mma = 0
            while tile_scheduler.valid():
                m_st, n_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M), Tx.meta_var(tile_scheduler.n_idx * BLK_N)

                # ── TMA async load: GMEM → SMEM (hardware-driven) ──
                @Tx.inline
                def tma_load(m_st, n_st, k_st, stage):
                    byte_count = (BLK_M * BLK_K + BLK_N * BLK_K) * 2

                    # NOTE: MBar now needs to point to the current stage
                    Tx.copy_async(Asmem[stage, :, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma_bar.ptr_to([stage]))
                    Tx.copy_async(Bsmem[stage, :, :], B[n_st : n_st + BLK_N, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma_bar.ptr_to([stage]))
                    Tx.ptx.mbarrier.arrive.expect_tx(tma_bar.ptr_to([stage]), byte_count)

                # ── MMA compute: wait for data, run tcgen05, wait for result ──
                @Tx.inline
                def mma(accum, stage):
                    # ---  Waits on tma_bar (data ready) ---
                    Tx.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)  # pyright: ignore[reportUnboundVariable]  # noqa: F823
                    if stage == PIPE_DEPTH - 1:  # phase flips when stage would wrap to zero
                        phase_tma ^= 1  # pyright: ignore[reportUnboundVariable]  # noqa: F841
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                    # --- Issues gemm_async + commit ---
                    # NOTE: for the current stage
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :], accum=accum, dispatch="tcgen05", cta_group=1)
                    Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)  # signal MMA done

                    # --- Waits on mma_bar (MMA done) ---
                    Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)  # pyright: ignore[reportUnboundVariable] # noqa: F823
                    phase_mma ^= 1  # pyright: ignore[reportUnboundVariable]  # noqa: F841

                # --- Main loop (elected thread of warp 0) ---
                with Tx.thread(parent="warpgroup")[thread_id == 0]:
                    # Prefetch PRE_NUM stages into SMEM
                    for pre_k in range(PRE_NUM):
                        stage = pre_k % PIPE_DEPTH
                        pre_k_st = pre_k * BLK_K

                        tma_load(m_st, n_st, pre_k_st, stage)

                    # For each K tile, wait for load to finish, compute, then issue the next load.
                    for k in range(K_TILES):
                        stage = k % PIPE_DEPTH

                        # Waits on load to finish, then compute the current tile
                        mma(k != 0, stage)

                        # Issue the load for the future tile into the newly freed stage
                        next_k = k + PRE_NUM
                        if next_k < K_TILES:
                            tma_load(m_st, n_st, next_k * BLK_K, stage)

                # SYNC RULE: MMA writes TMEM -> Threads read TMEM
                Tx.cuda.warpgroup_sync(10)
                Tx.ptx.tcgen05.fence.after_thread_sync()

                # ── Writeback: TMEM → Reg → SMEM → GMEM ──
                Dreg_f16 = Tx.alloc_local((MMA_N,), d_type)

                for no in Tx.unroll(MMA_N // TMEM_LD_N):
                    no_st = Tx.meta_var(no * TMEM_LD_N)
                    Dreg = Tx.alloc_local((TMEM_LD_N,), acc_type)  # tmp register

                    # --- all threads in WG write TMEM → Reg ---
                    with Tx.warpgroup():
                        Dreg_wg = Dreg.view(MMA_N, TMEM_LD_N, layout=TileLayout(S[(MMA_N, TMEM_LD_N) : (1 @ TLane, 1 @ TCol)]))
                        Tx.copy(Dreg_wg[:, :], tmem[:, no_st : no_st + TMEM_LD_N])

                    # -- per-thread cast ---
                    with Tx.thread():
                        Tx.cast(Dreg_f16[no_st : no_st + TMEM_LD_N], Dreg[:])

                for no in Tx.unroll(MMA_N // EPI_N):
                    no_st = Tx.meta_var(no * EPI_N)
                    n_epi_st = Tx.meta_var(n_st + no_st)

                    # --- per-thread write EPI_N-col slice to SMEM ---
                    with Tx.thread():
                        Tx.copy(Dsmem[thread_id, :], Dreg_f16[no_st : no_st + EPI_N])

                        # make SMEM writes visible to TMA engine
                        Tx.ptx.fence.proxy_async("shared::cta")

                    # wait until all threads done writing SMEM
                    Tx.cuda.warpgroup_sync(10)

                    # --- TMA Store (elected thread of warp 0) ---
                    with Tx.thread(parent="warpgroup")[thread_id == 0]:
                        Tx.copy_async(D[m_st : m_st + BLK_M, n_epi_st : n_epi_st + EPI_N], Dsmem[:, :], dispatch="tma")

                        # Commit and wait for TMA store completion
                        Tx.ptx.cp_async.bulk.commit_group()
                        Tx.ptx.cp_async.bulk.wait_group(0)

                    # Sync before next iteration
                    Tx.cuda.warpgroup_sync(10)

                Tx.cuda.cta_sync()

                tile_scheduler.next_tile()

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 7: Warp specialization (PIPE_DEPTH=2)
#   WG1: warp0 (MMA) + warp3 (TMA producer)
#   WG0: writeback (TMEM → RF → SMEM → GMEM)
#   4 barrier types: tma2mma, mma2tma, mma2ld, ld2mma
#   PIPE_DEPTH=2 (same as step 6, focus on warp spec structure)
# ======================================================================
# MARK: Step 7
def hgemm_v7(M, N, K):

    MMA_N = BLK_N
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    EPI_N = 64  # Optional, can be any value that divides MMA_N (e.g., 64, 128)
    TMEM_LD_N = 8  # Optional, can be any value that divides MMA_N (e.g., 8, 16, 128)
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    CTA_MASK = 1  # NOTE: use cta_mask=1 for TCGen05Bar.arrive (non-cluster kernel).

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            tile_scheduler = ClusterPersistentScheduler2D("ts", num_m_tiles=M // 128, num_n_tiles=N // 128, l2_group_size=8, num_clusters=SM_COUNT)
            tile_scheduler.init(bx)

            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")
            thread_id = Tx.meta_var(warp_id * THREADS_PER_WARP + lane_id)

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")  # TMA tells MMA when it's done loading
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")  # MMA tells MMA when it needs more data
            mma2ld = TCGen05Bar(pool, 1, "mma2tma")  # MMA tells writeback it's done
            ld2mma = MBarrier(pool, 1, "ld2mma")  # writeback tells MMA the SMEM is free to be used again

            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            # pool.move_base_to(1024)  # NOTE: Reset the base to 1024 so Dsmem reuses Asmem/Bsmem space.
            Dsmem = pool.alloc((BLK_M, EPI_N), d_type, layout=D_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if wg_id == 0:
                if warp_id == 0:
                    if lane_id == 0:
                        tma2mma.init(1)
                        mma2tma.init(1)
                        mma2ld.init(1)
                        ld2mma.init(128)  # needs to wait for all 128 threads in WG0 to finish writeback

                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            # --- TMA Producer (WG1/warp3) ---
            if wg_id == 1:
                if warp_id == 3:
                    tma_phase = PipelineState("tma", PIPE_DEPTH)
                    tma_phase.init(is_producer=True)

                    byte_count = (BLK_M * BLK_K + BLK_N * BLK_K) * 2

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            m_st, n_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M), Tx.meta_var(tile_scheduler.n_idx * BLK_N)
                            for k in Tx.serial(K_TILES):
                                k_st = k * BLK_K

                                # --- wait for MMA to signal TMA that it's not using SMEM ---
                                mma2tma.wait(tma_phase.stage, tma_phase.phase)

                                Tx.copy_async(Asmem[tma_phase.stage, :, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma2mma.ptr_to([tma_phase.stage]))
                                Tx.copy_async(Bsmem[tma_phase.stage, :, :], B[n_st : n_st + BLK_N, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma2mma.ptr_to([tma_phase.stage]))

                                # --- TMA signals MMA that data has been loaded from GMEM into SMEM ---
                                tma2mma.arrive(tma_phase.stage, byte_count)
                                tma_phase.move_to_next_stage()  # advance from loading stage to blocking stage

                            tile_scheduler.next_tile()

                # --- MMA Consumer (WG1/warp0) ---
                elif warp_id == 0:
                    mma_phase = PipelineState("mma", PIPE_DEPTH)
                    mma_phase.init(is_producer=False)
                    ld_phase = PipelineState("ld", 1)
                    ld_phase.init(is_producer=True)

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            # --- wait for Writeback to signal to MMA that TMEM has been successfully written to GMEM ---
                            ld2mma.wait(0, ld_phase.phase)

                            # now we can use TMEM for GEMM!
                            ld_phase.move_to_next_stage()  # move from writeback stage to blocking stage
                            accum = 0  # False initially, overwrites TMEM.

                            for k in Tx.serial(K_TILES):
                                # --- wait for TMA to signal to MMA that data has been loaded into SMEM
                                tma2mma.wait(mma_phase.stage, mma_phase.phase)

                                Tx.gemm_async(tmem[:, :BLK_N], Asmem[mma_phase.stage, :, :], Bsmem[mma_phase.stage, :, :], accum=accum, dispatch="tcgen05", cta_group=1)  # noqa F821 # type: ignore
                                accum = 1  # For all subsequent iterations, becomes True, and accumulates onto TMEM

                                # --- MMA signals to TMA that SMEM has been consumed, and is now free for another load
                                mma2tma.arrive(mma_phase.stage, cta_group=1, cta_mask=CTA_MASK)
                                mma_phase.move_to_next_stage()  # move from GEMM stage to blocking stage

                            # --- MMA signals to Writeback that TMEM has been filled and should now be sent back to GMEM ---
                            mma2ld.arrive(0, cta_group=1, cta_mask=CTA_MASK)

                            tile_scheduler.next_tile()

            # --- Writeback (WG0) ---
            elif wg_id == 0:
                wb_phase = PipelineState("wb", 1)
                wb_phase.init(is_producer=False)

                while tile_scheduler.valid():
                    m_st, n_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M), Tx.meta_var(tile_scheduler.n_idx * BLK_N)

                    # --- wait for MMA to signal to Writeback that TMEM has been filled and should now be sent back to GMEM ---
                    mma2ld.wait(0, wb_phase.phase)

                    # now we can write out our TMEM!
                    wb_phase.move_to_next_stage()
                    Tx.ptx.tcgen05.fence.after_thread_sync()  # make TMEM visible to TMA

                    # ── Writeback: TMEM → Reg → SMEM → GMEM ──
                    # use a temporary register and write back in n=TMEM_LD_N chunks
                    Dreg_f16 = Tx.alloc_local((MMA_N,), d_type)

                    for no in Tx.unroll(MMA_N // TMEM_LD_N):
                        no_st = Tx.meta_var(no * TMEM_LD_N)
                        Dreg = Tx.alloc_local((TMEM_LD_N,), acc_type)  # tmp register

                        with Tx.warpgroup():  # all threads in WG write TMEM → Reg
                            Dreg_wg = Dreg.view(MMA_N, TMEM_LD_N, layout=TileLayout(S[(MMA_N, TMEM_LD_N) : (1 @ TLane, 1 @ TCol)]))
                            Tx.copy(Dreg_wg[:, :], tmem[:, no_st : no_st + TMEM_LD_N])

                        with Tx.thread():  # per-thread cast
                            Tx.cast(Dreg_f16[no_st : no_st + TMEM_LD_N], Dreg[:])

                    for no in Tx.unroll(MMA_N // EPI_N):
                        no_st = Tx.meta_var(no * EPI_N)
                        n_epi_st = Tx.meta_var(n_st + no_st)

                        with Tx.thread():  # per-thread write EPI_N-col slice to SMEM
                            Tx.copy(Dsmem[thread_id, :], Dreg_f16[no_st : no_st + EPI_N])
                            Tx.ptx.fence.proxy_async("shared::cta")  # make SMEM writes visible to TMA engine

                        Tx.cuda.warpgroup_sync(10)  # wait until all threads done writing SMEM

                        # --- TMA Store (elected thread of warp 0) ---
                        with Tx.thread(parent="warpgroup")[thread_id == 0]:
                            Tx.copy_async(D[m_st : m_st + BLK_M, n_epi_st : n_epi_st + EPI_N], Dsmem[:, :], dispatch="tma")
                            # Commit and wait for TMA store completion
                            # NOTE: must stay in this loop to prevent SMEM overwrites in DMEM
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)

                        # Sync before next iteration
                        Tx.cuda.warpgroup_sync(10)

                    # --- Writeback signals to MMA that all threads done writing from TMEM to reg, let MMA know that TMEM is free ---
                    ld2mma.arrive(0, cta_id=0, pred=True)

                    tile_scheduler.next_tile()

            Tx.cuda.cta_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 8: Deeper pipeline (PIPE_DEPTH=4)
#   Same warp-specialized structure as v7, but with 4-stage pipeline
#   to better hide TMA latency. Only changes: PIPE_DEPTH=2 → 4,
#   which affects barrier array sizes and Asmem/Bsmem stage dimensions.
# ======================================================================
# MARK: Step 8
def hgemm_v8(M, N, K):
    MMA_N = BLK_N
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 8
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    CTA_MASK = 1  # NOTE: use cta_mask=1 for TCGen05Bar.arrive (non-cluster kernel).

    # Same structure as step 7 but with PIPE_DEPTH=4.
    # Changes needed:
    #   - TMABar(pool, 4, ...), TCGen05Bar(pool, 4, ...)
    #   - Asmem/Bsmem shape: (4, BLK_M, BLK_K) / (4, BLK_N, BLK_K)
    #   - PipelineState("tma", 4), PipelineState("mma", 4)
    # Everything else stays the same as step 7.
    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            tile_scheduler = ClusterPersistentScheduler2D("ts", num_m_tiles=M // 128, num_n_tiles=N // 128, l2_group_size=8, num_clusters=SM_COUNT)
            tile_scheduler.init(bx)

            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")
            thread_id = Tx.meta_var(warp_id * THREADS_PER_WARP + lane_id)

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")  # TMA tells MMA when it's done loading
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")  # MMA tells MMA when it needs more data
            mma2ld = TCGen05Bar(pool, 1, "mma2tma")  # MMA tells writeback it's done
            ld2mma = MBarrier(pool, 1, "ld2mma")  # writeback tells MMA the SMEM is free to be used again

            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            # pool.move_base_to(1024)  # NOTE: Reset the base to 1024 so Dsmem reuses Asmem/Bsmem space.
            Dsmem = pool.alloc((BLK_M, EPI_N), d_type, layout=D_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if wg_id == 0:
                if warp_id == 0:
                    if lane_id == 0:
                        tma2mma.init(1)
                        mma2tma.init(1)
                        mma2ld.init(1)
                        ld2mma.init(128)  # needs to wait for all 128 threads in WG0 to finish writeback

                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            # --- TMA Producer (WG1/warp3) ---
            if wg_id == 1:
                if warp_id == 3:
                    tma_phase = PipelineState("tma", PIPE_DEPTH)
                    tma_phase.init(is_producer=True)

                    byte_count = (BLK_M * BLK_K + BLK_N * BLK_K) * 2

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            m_st, n_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M), Tx.meta_var(tile_scheduler.n_idx * BLK_N)
                            for k in Tx.serial(K_TILES):
                                k_st = k * BLK_K

                                # --- wait for MMA to signal TMA that it's not using SMEM ---
                                mma2tma.wait(tma_phase.stage, tma_phase.phase)

                                Tx.copy_async(Asmem[tma_phase.stage, :, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma2mma.ptr_to([tma_phase.stage]))
                                Tx.copy_async(Bsmem[tma_phase.stage, :, :], B[n_st : n_st + BLK_N, k_st : k_st + BLK_K], dispatch="tma", cta_group=1, mbar=tma2mma.ptr_to([tma_phase.stage]))

                                # --- TMA signals MMA that data has been loaded from GMEM into SMEM ---
                                tma2mma.arrive(tma_phase.stage, byte_count)
                                tma_phase.move_to_next_stage()  # advance from loading stage to blocking stage

                            tile_scheduler.next_tile()

                # --- MMA Consumer (WG1/warp0) ---
                elif warp_id == 0:
                    mma_phase = PipelineState("mma", PIPE_DEPTH)
                    mma_phase.init(is_producer=False)
                    ld_phase = PipelineState("ld", 1)
                    ld_phase.init(is_producer=True)

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            # --- wait for Writeback to signal to MMA that TMEM has been successfully written to GMEM ---
                            ld2mma.wait(0, ld_phase.phase)

                            # now we can use TMEM for GEMM!
                            ld_phase.move_to_next_stage()  # move from writeback stage to blocking stage
                            accum = 0  # False initially, overwrites TMEM.

                            for k in Tx.serial(K_TILES):
                                # --- wait for TMA to signal to MMA that data has been loaded into SMEM
                                tma2mma.wait(mma_phase.stage, mma_phase.phase)

                                Tx.gemm_async(tmem[:, :BLK_N], Asmem[mma_phase.stage, :, :], Bsmem[mma_phase.stage, :, :], accum=accum, dispatch="tcgen05", cta_group=1)  # noqa F821 # type: ignore
                                accum = 1  # For all subsequent iterations, becomes True, and accumulates onto TMEM

                                # --- MMA signals to TMA that SMEM has been consumed, and is now free for another load
                                mma2tma.arrive(mma_phase.stage, cta_group=1, cta_mask=CTA_MASK)
                                mma_phase.move_to_next_stage()  # move from GEMM stage to blocking stage

                            # --- MMA signals to Writeback that TMEM has been filled and should now be sent back to GMEM ---
                            mma2ld.arrive(0, cta_group=1, cta_mask=CTA_MASK)

                            tile_scheduler.next_tile()

            # --- Writeback (WG0) ---
            elif wg_id == 0:
                wb_phase = PipelineState("wb", 1)
                wb_phase.init(is_producer=False)

                while tile_scheduler.valid():
                    m_st, n_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M), Tx.meta_var(tile_scheduler.n_idx * BLK_N)

                    # --- wait for MMA to signal to Writeback that TMEM has been filled and should now be sent back to GMEM ---
                    mma2ld.wait(0, wb_phase.phase)

                    # now we can write out our TMEM!
                    wb_phase.move_to_next_stage()
                    Tx.ptx.tcgen05.fence.after_thread_sync()  # make TMEM visible to TMA

                    # ── Writeback: TMEM → Reg → SMEM → GMEM ──
                    # use a temporary register and write back in n=TMEM_LD_N chunks
                    Dreg_f16 = Tx.alloc_local((MMA_N,), d_type)

                    for no in Tx.unroll(MMA_N // TMEM_LD_N):
                        no_st = Tx.meta_var(no * TMEM_LD_N)
                        Dreg = Tx.alloc_local((TMEM_LD_N,), acc_type)  # tmp register

                        with Tx.warpgroup():  # all threads in WG write TMEM → Reg
                            Dreg_wg = Dreg.view(MMA_N, TMEM_LD_N, layout=TileLayout(S[(MMA_N, TMEM_LD_N) : (1 @ TLane, 1 @ TCol)]))
                            Tx.copy(Dreg_wg[:, :], tmem[:, no_st : no_st + TMEM_LD_N])

                        with Tx.thread():  # per-thread cast
                            Tx.cast(Dreg_f16[no_st : no_st + TMEM_LD_N], Dreg[:])

                    for no in Tx.unroll(MMA_N // EPI_N):
                        no_st = Tx.meta_var(no * EPI_N)
                        n_epi_st = Tx.meta_var(n_st + no_st)

                        with Tx.thread():  # per-thread write EPI_N-col slice to SMEM
                            Tx.copy(Dsmem[thread_id, :], Dreg_f16[no_st : no_st + EPI_N])
                            Tx.ptx.fence.proxy_async("shared::cta")  # make SMEM writes visible to TMA engine

                        Tx.cuda.warpgroup_sync(10)  # wait until all threads done writing SMEM

                        # --- TMA Store (elected thread of warp 0) ---
                        with Tx.thread(parent="warpgroup")[thread_id == 0]:
                            Tx.copy_async(D[m_st : m_st + BLK_M, n_epi_st : n_epi_st + EPI_N], Dsmem[:, :], dispatch="tma")
                            # Commit and wait for TMA store completion
                            # NOTE: must stay in this loop to prevent SMEM overwrites in DMEM
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)

                        # Sync before next iteration
                        Tx.cuda.warpgroup_sync(10)

                    # --- Writeback signals to MMA that all threads done writing from TMEM to reg, let MMA know that TMEM is free ---
                    ld2mma.arrive(0, cta_id=0, pred=True)

                    tile_scheduler.next_tile()

            Tx.cuda.cta_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 9: Cluster — 2-CTA cooperation
#   CTA_GROUP=2, MMA_M=MMA_N=256, cross-CTA TMEM sharing.
# ======================================================================
# MARK: Step 9
def hgemm_v9(M, N, K):
    CTA_GROUP = 2

    # MMA output is MMA_N=256 columns (B_N * CTA_GROUP)
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 8
    WG_NUMBER = 2
    DTYPE_SIZE = a_type.bits // 8

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    CTA_MASK = 3  # NOTE: use cta_mask=3 for TCGen05Bar.arrive (signal both CTAs)

    # Extend step 7 with CTA_GROUP=2 cluster.
    #   - Use cluster_sync instead of cta_sync at boundaries
    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")
            bx = Tx.cta_id([SM_COUNT], parent="kernel")

            tile_scheduler = ClusterPersistentScheduler2D("ts", num_m_tiles=M // MMA_N, num_n_tiles=N // MMA_N, l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
            tile_scheduler.init(bx // CTA_GROUP)

            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([WARPS_PER_WG], parent="warpgroup")
            lane_id = Tx.thread_id([THREADS_PER_WARP], parent="warp")
            thread_id = Tx.meta_var(warp_id * THREADS_PER_WARP + lane_id)

            # --- Shared memory allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")  # TMA tells MMA when it's done loading
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")  # MMA tells MMA when it needs more data
            mma2ld = TCGen05Bar(pool, 1, "mma2tma")  # MMA tells writeback it's done
            ld2mma = MBarrier(pool, 1, "ld2mma")  # writeback tells MMA the SMEM is free to be used again

            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            # pool.move_base_to(1024)  # NOTE: Reset the base to 1024 so Dsmem reuses Asmem/Bsmem space.
            Dsmem = pool.alloc((BLK_M, EPI_N), d_type, layout=D_layout)
            pool.commit()

            # --- Barrier + TMEM init (warp 0 only) ---
            if wg_id == 0:
                if warp_id == 0:
                    if lane_id == 0:
                        tma2mma.init(1)
                        mma2tma.init(1)
                        mma2ld.init(1)
                        # ld2mma.init(128 * CTA_GROUP) for cross-CTA writeback sync
                        ld2mma.init(128 * CTA_GROUP)  # CHANGED: was 128

                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cluster_sync()

            tmem = Tx.decl_buffer((TMEM_LANES, 512), acc_type, scope="tmem", allocated_addr=0, layout=TileLayout(S[(TMEM_LANES, 512) : (1 @ TLane, 1 @ TCol)]))

            # --- TMA Producer (WG1/warp3) ---
            if wg_id == 1:
                if warp_id == 3:
                    # tma2mma_cta0 = tma2mma.remote_view(0) for crss-CTA signaling
                    tma2mma_cta0 = tma2mma.remote_view(0)  # NEW: cross-CTA barrier view

                    tma_phase = PipelineState("tma", PIPE_DEPTH)
                    tma_phase.init(is_producer=True)

                    byte_count = CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            m_st, n_st = Tx.meta_var((tile_scheduler.m_idx + cbx) * BLK_M), Tx.meta_var((tile_scheduler.m_idx + cbx) * BLK_N)
                            for k in Tx.serial(K_TILES):
                                k_st = k * BLK_K

                                # --- wait for MMA to signal TMA that it's not using SMEM ---
                                mma2tma.wait(tma_phase.stage, tma_phase.phase)

                                Tx.copy_async(
                                    Asmem[tma_phase.stage, :, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K], dispatch="tma", cta_group=CTA_GROUP, mbar=tma2mma_cta0.ptr_to([tma_phase.stage])
                                )
                                Tx.copy_async(
                                    Bsmem[tma_phase.stage, :, :], B[n_st : n_st + BLK_N, k_st : k_st + BLK_K], dispatch="tma", cta_group=CTA_GROUP, mbar=tma2mma_cta0.ptr_to([tma_phase.stage])
                                )

                                # --- TMA signals MMA that data has been loaded from GMEM into SMEM ---
                                if cbx == 0:
                                    tma2mma.arrive(tma_phase.stage, byte_count)
                                    tma_phase.move_to_next_stage()  # advance from loading stage to blocking stage

                            tile_scheduler.next_tile()

                # --- MMA Consumer (WG1/warp0) ---
                elif warp_id == 0:
                    # MMA only on cbx==0 (CTA 0 issues MMA for both CTAs)
                    if cbx == 0:
                        mma_phase = PipelineState("mma", PIPE_DEPTH)
                        mma_phase.init(is_producer=False)
                        ld_phase = PipelineState("ld", 1)
                        ld_phase.init(is_producer=True)

                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            while tile_scheduler.valid():
                                # --- wait for Writeback to signal to MMA that TMEM has been successfully written to GMEM ---
                                ld2mma.wait(0, ld_phase.phase)

                                # now we can use TMEM for GEMM!
                                ld_phase.move_to_next_stage()  # move from writeback stage to blocking stage
                                accum = 0  # False initially, overwrites TMEM.

                                for k in Tx.serial(K_TILES):
                                    # --- wait for TMA to signal to MMA that data has been loaded into SMEM
                                    tma2mma.wait(mma_phase.stage, mma_phase.phase)

                                    Tx.gemm_async(tmem[:, :MMA_N], Asmem[mma_phase.stage, :, :], Bsmem[mma_phase.stage, :, :], accum=accum, dispatch="tcgen05", cta_group=CTA_GROUP)  # noqa F821 # type: ignore
                                    accum = 1  # For all subsequent iterations, becomes True, and accumulates onto TMEM

                                    # --- MMA signals to TMA that SMEM has been consumed, and is now free for another load
                                    mma2tma.arrive(mma_phase.stage, cta_group=CTA_GROUP, cta_mask=CTA_MASK)
                                    mma_phase.move_to_next_stage()  # move from GEMM stage to blocking stage

                                # --- MMA signals to Writeback that TMEM has been filled and should now be sent back to GMEM ---
                                mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=CTA_MASK)

                                tile_scheduler.next_tile()

            # --- Writeback (WG0) ---
            elif wg_id == 0:
                wb_phase = PipelineState("wb", 1)
                wb_phase.init(is_producer=False)

                while tile_scheduler.valid():
                    m_st, n_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M), Tx.meta_var(tile_scheduler.n_idx * BLK_N)

                    # --- wait for MMA to signal to Writeback that TMEM has been filled and should now be sent back to GMEM ---
                    mma2ld.wait(0, wb_phase.phase)

                    # now we can write out our TMEM!
                    wb_phase.move_to_next_stage()
                    Tx.ptx.tcgen05.fence.after_thread_sync()  # make TMEM visible to TMA

                    # ── Writeback: TMEM → Reg → SMEM → GMEM ──
                    # use a temporary register and write back in n=TMEM_LD_N chunks
                    Dreg_f16 = Tx.alloc_local((MMA_N,), d_type)

                    for no in Tx.unroll(MMA_N // TMEM_LD_N):
                        no_st = Tx.meta_var(no * TMEM_LD_N)
                        Dreg = Tx.alloc_local((TMEM_LD_N,), acc_type)  # tmp register

                        with Tx.warpgroup():  # all threads in WG write TMEM → Reg
                            Dreg_wg = Dreg.view(MMA_N, TMEM_LD_N, layout=TileLayout(S[(MMA_N, TMEM_LD_N) : (1 @ TLane, 1 @ TCol)]))
                            Tx.copy(Dreg_wg[:, :], tmem[:, no_st : no_st + TMEM_LD_N])

                        with Tx.thread():  # per-thread cast
                            Tx.cast(Dreg_f16[no_st : no_st + TMEM_LD_N], Dreg[:])

                    for no in Tx.unroll(MMA_N // EPI_N):
                        no_st = Tx.meta_var(no * EPI_N)
                        n_epi_st = Tx.meta_var(n_st + no_st)

                        with Tx.thread():  # per-thread write EPI_N-col slice to SMEM
                            Tx.copy(Dsmem[thread_id, :], Dreg_f16[no_st : no_st + EPI_N])
                            Tx.ptx.fence.proxy_async("shared::cta")  # make SMEM writes visible to TMA engine

                        Tx.cuda.warpgroup_sync(10)  # wait until all threads done writing SMEM

                        # --- TMA Store (elected thread of warp 0) ---
                        with Tx.thread(parent="warpgroup")[thread_id == 0]:
                            Tx.copy_async(D[m_st : m_st + BLK_M, n_epi_st : n_epi_st + EPI_N], Dsmem[:, :], dispatch="tma")
                            # Commit and wait for TMA store completion
                            # NOTE: must stay in this loop to prevent SMEM overwrites in DMEM
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)

                        # Sync before next iteration
                        Tx.cuda.warpgroup_sync(10)

                    # --- Writeback signals to MMA that all threads done writing from TMEM to reg, let MMA know that TMEM is free ---
                    ld2mma.arrive(0, cta_id=0, pred=True)

                    tile_scheduler.next_tile()

            Tx.cuda.cluster_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel


# ======================================================================
# Step 10: 2-consumer warp specialization
#   NUM_CONSUMER=2, WG2 (TMA+MMA), WG0/WG1 (writeback).
#   This is the final optimized kernel.
# ======================================================================
# MARK: Step 10
def hgemm_v10(M, N, K):
    CTA_GROUP = 2
    NUM_CONSUMER = 2
    MMA_M, MMA_N, MMA_K = 256, 256, 16
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 8
    WG_NUMBER = 3
    DTYPE_SIZE = a_type.bits // 8

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N))

    CTA_MASK = 3  # NOTE: use cta_mask=3 for TCGen05Bar.arrive (signal both CTAs)

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            # TODO: 3 warpgroups, 2 consumers, 2-CTA cluster.
            # Key changes from step 9:
            #   - WG_NUMBER=3: WG2 (TMA+MMA), WG0+WG1 (writeback)
            #   - NUM_CONSUMER=2 MMA warps (warp0, warp1 in WG2)
            #   - Each MMA warp handles tmem[:, warp_id*MMA_N : warp_id*MMA_N+MMA_N]
            #   - TMA loads NUM_CONSUMER A blocks per stage
            #   - mma2tma.init(NUM_CONSUMER), mma2ld depth=NUM_CONSUMER
            #   - WG0/WG1 read from tmem offset by wg_id*MMA_N
            #   - Writeback uses per-consumer Dsmem[wg_id, ...]
            pass

    return kernel
