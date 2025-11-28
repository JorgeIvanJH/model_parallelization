This repo explores different python concurrency techniques (with and without paralelization), and allows for testing its performance on a specified task, more specificly inference of different ML models on datasets.

[src/concurrency](src/concurrency) covers implementation of multiple concurrency techniques on a heavy task

[src/ml](src/ml) contains basic implementation and training of different ML models

[src/ml_concurrency](src/ml_concurrency) test and records performance of the concurrency methods on inference of trained (and saved models from [src/ml](src/ml)) over datasets stored in [data](data)



## Speedup (heavy task w built-in)
    Threads: ~6.5x (Python 3.14 NoGil, No Support for Libraries yet)
    Process: ~4.8x (Python 3.x)
    Pool: ~3.5 (Python 3.x)
    ProcessPoolExecutor: ~4.2 (Python >= 3.2)

## Findings
    Older versions of python are generally slower, both on sequential and on multiprocessing

    Async processes are always a bit faster

    Each process has its own memory space:
        - Large datasets are duplicated across processes
        - Use shared memory for large arrays (numpy with shared_memory)
        - Monitor memory usage to avoid system slowdown

    Creating Processes has overhead:
        - batch small tasks together
        - use multiprocessing for tasks taking > 0.1s

## Challenges
    We will likely have to do multiprocessing for true parallelism, Threading not possible for parallelism due to lack of support for many libraries.
        On Windows: (problem)
            uses "spawn" method
            Processes created with copy of the data. Each process will have a copy of the data on memory, easily consuming all the memory
        On Linux: (possible solution)
            uses "fork" method
            Shared memory, no true copy of the data. Less memory strain. 
    However, if I/O takes time, multithreading will be a fairly good choice
    # Of processes (workers)
        usual good default:
            num_workers = os.cpu_count()
        more if:
            short tasks (<2ms per task)
        less if:
            processes are memory heavy
            worker startup is large

## TODOs
    try start method "fork" and "forkserver" on docker (are faster than Windows' "spawn") but only works on linux environments
    define clear unit test to avoid problems with multithreads and ensure data consistency, regardless of the configuration established
    See Model's built-in multithreading
    Compare memory overhead between running on Linux (fork), and windows (spawn)