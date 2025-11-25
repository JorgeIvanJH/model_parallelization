## Speedup (heavy task w built-in)
    Threads: ~6.5x (Python 3.14 NoGil, No Support for Libraries yet)
    Process: ~4.8x (Python 3.x)
    Pool: ~3.5 (Python 3.x)

## Findings
    Older versions of python are generally slower, both on sequential and on multiprocessing

    Each process has its own memory space:
        - Large datasets are duplicated across processes
        - Use shared memory for large arrays (numpy with shared_memory)
        - Monitor memory usage to avoid system slowdown

    Creating Processes has overhead:
        - batch small tasks together
        - use multiprocessing for tasks taking > 0.1s