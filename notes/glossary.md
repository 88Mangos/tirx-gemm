```python
"""
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
```



```python
""" 
MMA_N     : output tile width 
EPI_N     : epilogue width. This is a TMA Store Constraint. 
            Why 64? Blackwell's TMA engine and SMEM layouts (like SWIZZLE_128B_ATOM) often perform best when writing back data in specific power-of-two widths.
            In your code, you process the 128-wide tile in two 64-column "slices" to fit into the Dsmem buffer you allocated. 
            It prevents you from needing a massive 128x128 SMEM buffer for the output, saving shared memory.
            
            # Optional, can be any value that divides MMA_N (e.g., 64, 128)

TMEM_LD_N : Register Load Width. This is the Thread-Level Granularity.
            When moving data TMEM → Registers, the warpgroup (128 threads) reads a small "strip" of columns.
            8 is chosen because each thread can easily hold 8 float32 values in its local registers (Dreg = Tx.alloc_local((8,), "float32")).
            If you increased this to 16, each thread would need more registers, potentially hitting "Register Pressure" limits which slows down the GPU.

            # Optional, can be any value that divides MMA_N (e.g., 8, 16, 128)
"""
```