### Example: Correct Step 7 Structure

A correctly compiled Step 7 kernel has this structure (from the reference implementation):

```c
// ---- Barrier inits: threadIdx.x < 1 guard (CTA thread 0 only) ----
if (((int)threadIdx.x) < 1) {
  tvm_builtin_ptx_mbarrier_init(&s_buf[4], 1);   // tma2mma slot 0
  tvm_builtin_ptx_mbarrier_init(&s_buf[5], 1);   // tma2mma slot 1
}
if (((int)threadIdx.x) < 1) {
  tvm_builtin_ptx_mbarrier_init(&s_buf[6], 1);   // mma2tma slot 0
  tvm_builtin_ptx_mbarrier_init(&s_buf[7], 1);   // mma2tma slot 1
}
if (((int)threadIdx.x) < 1) {
  tvm_builtin_ptx_mbarrier_init(&s_buf[8], 1);   // mma2ld
}
if (((int)threadIdx.x) < 1) {
  tvm_builtin_ptx_mbarrier_init(&s_buf[9], 128); // ld2mma
}

// ---- TMEM alloc: WG0 warp0 (all 32 lanes, no lane guard) ----
if ((warp_id_in_cta >> 2) == 0) {       // wg_id == 0
  if ((warp_id_in_cta % 4) == 0) {      // warp_id == 0
    tvm_builtin_ptx_tcgen05_alloc_cta_group_1(&s_buf[0], 512);
  }
}

// ---- Fences + sync ----
tvm_builtin_ptx_fence_proxy_async_shared_cta();
tvm_builtin_ptx_fence_mbarrier_init();
tvm_builtin_cuda_cta_sync();

// ---- Pipeline phase init ----
tma_phase_ptr[0] = 1;     // producer: starts at phase 1 (first wait passes)
mma_phase_ptr[0] = 0;     // consumer: starts at phase 0 (first wait blocks)
ld_phase_ptr[0] = 1;      // producer
wb_phase_ptr[0] = 0;      // consumer

// ---- WG1: TMA warp (warp 3) ----
if ((warp_id_in_cta >> 2) == 1) {       // wg_id == 1
  if ((warp_id_in_cta & 3) == 3) {      // warp_id == 3
    if (tvm_builtin_elect_one_sync_op()) {
      while (valid) {
        for (int k = 0; k < 16; ++k) {  // K_TILES iterations
          mbarrier_wait(mma2tma[stage], phase);   // wait for SMEM free
          cp_async_bulk_tensor(Asmem[stage], ...); // TMA load A
          cp_async_bulk_tensor(Bsmem[stage], ...); // TMA load B
          mbarrier_arrive_expect_tx(tma2mma[stage], 32768); // signal MMA
          stage = (stage + 1) % 2;
          if (stage == 0) phase ^= 1;
        }
        next_tile();
      }
    }
  } else {
    // ---- WG1: MMA warp (warp 0) ----
    if ((warp_id_in_cta % 4) == 0) {    // warp_id == 0
      if (tvm_builtin_elect_one_sync_op()) {
        while (valid) {
          mbarrier_wait(ld2mma, ld_phase);         // wait for TMEM free
          for (int k = 0; k < 16; ++k) {           // K_TILES iterations
            mbarrier_wait(tma2mma[stage], phase);   // wait for data
            tcgen05_fence_after_thread_sync();
            tcgen05_mma(...);                       // MMA
            tcgen05_commit(mma2tma[stage]);          // signal TMA
            stage = (stage + 1) % 2;
            if (stage == 0) phase ^= 1;
          }
          tcgen05_commit(mma2ld);                   // signal writeback
          next_tile();
        }
      }
    }
  }
} else {
  // ---- WG0: Writeback ----
  if ((warp_id_in_cta >> 2) == 0) {     // wg_id == 0
    while (valid) {
      mbarrier_wait(mma2ld, wb_phase);             // wait for MMA done
      tcgen05_ld(...);                              // read TMEM → registers
      mbarrier_arrive_remote(ld2mma, 0, true);     // signal MMA: TMEM free
      // ... cast, write to Dsmem, TMA store ...
      next_tile();
    }
  }
}

// ---- Cleanup ----
cta_sync();
if ((warp_id_in_cta % 4) == 0) {       // warp_id == 0 (all 32 lanes)
  tcgen05_relinquish_alloc_permit();
  tcgen05_dealloc(s_buf[0], 512);
}
```

