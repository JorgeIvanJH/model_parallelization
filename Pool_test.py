import multiprocessing as mp
from utils import cpu_intensive_task, sequential_execution, measure_time_decorator
from utils import TASK_COMPLEXITY, NUM_TASKS, NUM_WORKERS

@measure_time_decorator
def parallel_execution(num_tasks, task_complexity):
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(cpu_intensive_task, [task_complexity] * num_tasks)
    return results

if __name__ == '__main__':

    # Sequential execution
    results_seq , sequential_time = sequential_execution(NUM_TASKS, TASK_COMPLEXITY)
    print(f"Sequential execution: {sequential_time:.2f} seconds")
    
    # Parallel execution
    results_par, parallel_time = parallel_execution(NUM_TASKS, TASK_COMPLEXITY)
    
    print(f"Parallel execution: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    assert results_seq == results_par
    print("OK")
    print("results: ", results_par)