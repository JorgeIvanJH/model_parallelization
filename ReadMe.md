This repo explores different python concurrency techniques (with and without paralelization), and allows for testing its performance on a specified task, more specificly inference of different ML models on datasets.

## Structure

[src/concurrency](src/concurrency) covers implementation of multiple concurrency techniques on a heavy task
- [methods/](src/concurrency/methods) - implementation of different concurrency methods (Threads, Process, Pool, ProcessPoolExecutor)
- [testing/](src/concurrency/testing) - benchmarking scripts and performance results

[src/ml](src/ml) contains basic implementation and training of different ML models
- [training/](src/ml/training) - jupyter notebooks for model training
- [saved_models/](src/ml/saved_models) - serialized trained models (LogisticRegression, LightGBM)

[src/ml_concurrency](src/ml_concurrency) test and records performance of the concurrency methods on inference of trained (and saved models from [src/ml](src/ml)) over datasets stored in [data](data)
- [testing/](src/ml_concurrency/testing) - scripts to benchmark ML inference with concurrency
- [results/](src/ml_concurrency/testing/results) - performance metrics (runs.csv, EDA.ipynb)

[data](data) - healthcare datasets for testing
- Healthcare_Investments_and_Hospital_Stay.csv
- healthcare_noshows_appointments.csv

[environments](environments) - conda environment configurations
- python310.yml - Python 3.10 environment
- python314.yml - Python 3.14 standard environment
- python314nogil.yml - Python 3.14 free-threaded (no GIL) experimental build

[notebooks](notebooks) - exploratory testing notebooks

## Speedup (heavy task w built-in)
    Threads: ~10.4x (avg over 105 runs)
    ProcessPoolExecutor: ~5.6x (avg over 104 runs)
    Process: ~5.1x (avg over 104 runs)
    Pool: ~4.5x (avg over 104 runs)

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

    ML inference with multiprocessing:
        - Sequential is often faster for smaller datasets (< 1M rows)
        - Parallel overhead dominates when per-row inference is fast
        - Sweet spot around 4-7 workers for 10M rows on typical systems
        - Memory usage scales linearly with num_workers (2x-4x baseline)

## Challenges
    We will likely have to do multiprocessing for true parallelism, Threading not possible for parallelism due to lack of support for many libraries.
        On Windows: (problem)
            uses "spawn" method
            Processes created with copy of the data. Each process will have a copy of the data on memory, easily consuming all the memory
        On Linux: (possible solution)
            uses "fork" method
            Shared memory, no true copy of the data. Less memory strain. 
    However, if I/O takes time, multithreading will be a fairly good choice
    Python 3.14 NoGIL (free-threaded) build:
        - Experimental support, not production-ready
        - Many libraries not yet compatible (pandas, scikit-learn)
        - Theoretical performance gains for CPU-bound tasks with threads
    # Of processes (workers)
        usual good default:
            num_workers = os.cpu_count()
        more if:
            short tasks (<2ms per task)
        less if:
            processes are memory heavy
            worker startup is large

## TODOs
    CODE TO TEST NUM ROWS vs NUM CORES TO CHOOSE OPTIMAL
    try start method "fork" and "forkserver" on docker (are faster than Windows' "spawn") but only works on linux environments
    define clear unit test to avoid problems with multithreads and ensure data consistency, regardless of the configuration established
    See Model's built-in multithreading
    Compare memory overhead between running on Linux (fork), and windows (spawn)
    Test Python 3.14 free-threaded build once library support improves