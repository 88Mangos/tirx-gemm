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