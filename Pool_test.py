from multiprocessing import Pool
import time

from utils import cpu_intensive_task, sequential_execution
from utils import NUM_REPS, NUM_TASKS, NUM_WORKERS


def parallel_execution(num_tasks, num_reps):
    start = time.time()
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(cpu_intensive_task, [num_reps] * num_tasks)
    execution_time = time.time() - start
    return results, execution_time

if __name__ == '__main__':

    # Sequential execution
    results_seq , sequential_time = sequential_execution(NUM_TASKS, NUM_REPS)
    print(f"Sequential execution: {sequential_time:.2f} seconds")
    
    # Parallel execution
    results_par, parallel_time = parallel_execution(NUM_TASKS, NUM_REPS)
    print(f"Parallel execution: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    assert results_seq == results_par
    print("OK")
    print("results: ", results_par)