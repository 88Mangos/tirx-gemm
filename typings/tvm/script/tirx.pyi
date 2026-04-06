from typing import Any, ContextManager, Iterable, Union
from typings.tvm import DataType
import tvm

int32 = int
float32 = float
float16 = float
bfloat16 = float

# --- Add Buffer ---
class Buffer:
    """A multi-dimensional tensor in TVM TIR."""
    def __init__(self, shape: tuple, dtype: DataType | tvm.DataType): ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    @property
    def data(self) -> Any: ...

def prim_func(tirx: bool = True) -> Any:
    """
    Declare a TIRX primitive function.
    This is the entry point for defining a hardware-accelerated GPU kernel.
    """
    ...

class ScopeContextManager(ContextManager[None]):
    def __getitem__(self, condition: Any) -> "ScopeContextManager": ...

def kernel() -> ScopeContextManager:
    """Context manager defining the absolute scope of the GPU kernel execution."""
    ...

def cta() -> ScopeContextManager:
    """Scope: All threads in the Cooperative Thread Array (CTA) execute this block."""
    ...

def warpgroup() -> ScopeContextManager:
    """Scope: All 128 threads in the warpgroup execute this block."""
    ...

def warp() -> ScopeContextManager:
    """Scope: All 32 threads in the warp execute this block."""
    ...

def thread(parent: str = ...) -> ScopeContextManager:
    """Scope: Each thread executes independently."""
    ...

def cta_id(shape: list[int], parent: str = ...) -> Any:
    """Get the CTA (Block) index within the grid or cluster."""
    ...

def warpgroup_id(shape: list[int], parent: str = ...) -> Any:
    """Get the Warpgroup index (0-3) within the current CTA."""
    ...

def warp_id(shape: list[int], parent: str = ...) -> Any:
    """Get the Warp index within the current warpgroup."""
    ...

def thread_id(shape: list[int], parent: str = ...) -> Any:
    """Get the Thread (lane) index (0-31) within the current warp."""
    ...

class PoolAllocator:
    """Shared memory pool allocator."""
    def alloc(self, shape: tuple, dtype: Union[DataType, str], layout: Any = ..., align: int = 8) -> Any:
        """Allocate a buffer from the shared memory pool."""
        ...
    def move_base_to(self, offset: int) -> None:
        """Set the next allocation offset (useful for forcing alignments or overlapping buffers)."""
        ...
    def commit(self) -> None:
        """Finalize all shared memory allocations."""
        ...

def alloc_local(shape: tuple, dtype: DataType) -> Any:
    """Allocate a per-thread register buffer (Local Memory)."""
    ...

def decl_buffer(shape: tuple, dtype: DataType, scope: str = "tmem", **kwargs) -> Any:
    """Declare a Tensor Memory (TMEM) buffer."""
    ...

def address_of(buf: Any) -> Any:
    """Get the memory address of a buffer (commonly used for TMEM allocation)."""
    ...

def copy(dst: Any, src: Any) -> None:
    """Synchronous memory copy between buffers."""
    ...

def copy_async(dst: Any, src: Any, dispatch: str = "tma", **kwargs) -> None:
    """Asynchronous memory copy, typically using the Tensor Memory Accelerator (TMA)."""
    ...

def cast(dst: Any, src: Any) -> None:
    """Element-wise type cast (e.g., f32 to f16)."""
    ...

def gemm_async(C: Any, A: Any, B: Any, accum: bool, dispatch: str = "tcgen05", cta_group: int = 1) -> None:
    """
    Issue an asynchronous Matrix Multiply-Accumulate (MMA) instruction to the Tensor Cores.
    """
    ...

def unroll(N: Any) -> Iterable[int]:
    """Explicit unrolled loop."""
    ...

def serial(N: Any) -> Iterable[int]:
    """Sequential loop (not unrolled), represented as a TIR variable."""
    ...

def meta_var(expr: Any) -> Any:
    """Compile-time alias for an expression (required for buffer slice offsets)."""
    ...

def inline(func: Any) -> Any:
    """Decorator for inline helper functions within the kernel."""
    ...

# --- Namespaces --- Fix Namespaces with @staticmethod ---
class ptx:
    @staticmethod
    def elect_sync() -> Any:
        """Elect a single thread in a warp."""
        ...

    class tcgen05:
        @staticmethod
        def alloc(addr: Any, n_cols: int, cta_group: int) -> None:
            """Allocate Tensor Memory (TMEM) columns."""
            ...
        @staticmethod
        def relinquish_alloc_permit(cta_group: int) -> None:
            """Release the TMEM allocation permit."""
            ...
        @staticmethod
        def dealloc(addr: Any, n_cols: int, cta_group: int) -> None:
            """Deallocate Tensor Memory (TMEM)."""
            ...
        @staticmethod
        def commit(ptr: Any, cta_group: int, cta_mask: Any = ...) -> None:
            """Tell hardware to signal an mbarrier when MMA completes."""
            ...
        class fence:
            @staticmethod
            def after_thread_sync() -> None:
                """Fence required before accessing TMEM/SMEM after sync."""
                ...

    class mbarrier:
        @staticmethod
        def init(ptr: Any, count: int) -> None: 
            """ 
            mbarrier.init(ptr, count) Initializes a new MBarrier
            with expected_count=count, i.e., 
            the number of arrivals expected to this semaphore
            """
            ...
        @staticmethod
        def try_wait(ptr: Any, phase: Any) -> None: ...
        class arrive:
            @staticmethod
            def expect_tx(ptr: Any, bytes: int) -> None: ...

    class fence:
        @staticmethod
        def proxy_async(scope: str) -> None: ...
        @staticmethod
        def mbarrier_init() -> None: ...

    class cp_async:
        class bulk:
            @staticmethod
            def commit_group() -> None: ...
            @staticmethod
            def wait_group(n: int) -> None: ...

class cuda:
    @staticmethod
    def cta_sync() -> None: ...
    @staticmethod
    def warpgroup_sync(barrier_id: int) -> None: ...
    @staticmethod
    def cluster_sync() -> None: ...
