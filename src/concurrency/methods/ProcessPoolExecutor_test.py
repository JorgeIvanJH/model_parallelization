import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils import cpu_intensive_task, sequential_execution, measure_time_decorator, store_results
from src.utils import TASK_COMPLEXITY, NUM_TASKS, NUM_WORKERS, NUM_REPS, RESULTS_FILE
import random

@measure_time_decorator(times=NUM_REPS)
def async_parallel_execution(num_tasks, task_complexity):
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(cpu_intensive_task, task_complexity) for _ in range(num_tasks)]
        for future in as_completed(futures): # without "as_completed", the process is synchronous 
            results.append(future.result())
    return results

@measure_time_decorator(times=NUM_REPS)
def sync_parallel_execution(num_tasks, task_complexity):
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(cpu_intensive_task, [task_complexity] * num_tasks))
    return results

if __name__ == '__main__':

    # Sequential execution
    results_seq, sequential_time = sequential_execution(NUM_TASKS, TASK_COMPLEXITY)
    print(f"Sequential execution: {sequential_time:.2f} seconds")

    # Parallel execution
    results_par, parallel_time = async_parallel_execution(NUM_TASKS, TASK_COMPLEXITY)
    print(f"Parallel execution: {parallel_time:.2f} seconds")
    assert results_seq == results_par

    # Speedup
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")
    
    print("OK")
    
    # Store Results
    store_results(RESULTS_FILE, "ProcessPoolExecutor",sequential_time, parallel_time, speedup, NUM_REPS)


