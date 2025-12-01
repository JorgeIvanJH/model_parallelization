import multiprocessing as mp
from src.concurrency.utils import cpu_intensive_task, sequential_execution, measure_time_decorator, store_results
from src.concurrency.utils import TASK_COMPLEXITY, NUM_TASKS, NUM_WORKERS, NUM_REPS, RESULTS_FILE

@measure_time_decorator(times=NUM_REPS)
def parallel_execution(num_tasks, task_complexity):
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(cpu_intensive_task, [task_complexity] * num_tasks)
    return results

if __name__ == '__main__':

    # Sequential execution
    results_seq, sequential_time = sequential_execution(NUM_TASKS, TASK_COMPLEXITY)
    print(f"Sequential execution: {sequential_time:.2f} seconds")

    # Parallel execution
    results_par, parallel_time = parallel_execution(NUM_TASKS, TASK_COMPLEXITY)
    print(f"Parallel execution: {parallel_time:.2f} seconds")
    assert results_seq == results_par

    # Speedup
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")
    
    print("OK")
    
    # Store Results
    store_results(RESULTS_FILE, "Pool",sequential_time, parallel_time, speedup, NUM_REPS)


