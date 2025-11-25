from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from utils import cpu_intensive_task, sequential_execution, measure_time_decorator
from utils import TASK_COMPLEXITY, NUM_TASKS, NUM_WORKERS
import random

@measure_time_decorator
def async_parallel_execution(num_tasks, task_complexity):
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(cpu_intensive_task, task_complexity) for _ in range(num_tasks)]
        for future in as_completed(futures): # without "as_completed", the process is synchronous 
            results.append(future.result())
    return results

@measure_time_decorator
def sync_parallel_execution(num_tasks, task_complexity):
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(cpu_intensive_task, [task_complexity] * num_tasks))
    return results

if __name__ == '__main__':

    # Sequential execution
    results_seq , sequential_time = sequential_execution(NUM_TASKS, TASK_COMPLEXITY)
    print(f"Sequential execution: {sequential_time:.2f} seconds")
    
    # Parallel execution
    results_par, parallel_time = async_parallel_execution(NUM_TASKS, TASK_COMPLEXITY)
    
    print(f"async Parallel execution: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")

    results_par, parallel_time = sync_parallel_execution(NUM_TASKS, TASK_COMPLEXITY)
    print(f"sync Parallel execution: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")

    assert results_seq == results_par
    print("OK")
    print("results: ", results_par)