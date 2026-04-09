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
