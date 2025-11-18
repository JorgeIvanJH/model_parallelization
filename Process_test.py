import multiprocessing as mp
import time

from utils import cpu_intensive_task, sequential_execution, measure_time_decorator
from utils import TASK_COMPLEXITY, NUM_TASKS, START_METHOD

def worker(queue, task_complexity):
    result = cpu_intensive_task(task_complexity)
    queue.put(result)

@measure_time_decorator
def parallel_execution(num_tasks, task_complexity):

    queue = mp.Queue()
    processes = []
    for i in range(num_tasks):
        p = mp.Process(target=worker, args=(queue, task_complexity))
        processes.append(p)
        p.start()
    # Wait for all processes to complete
    for p in processes:
        p.join()
    results = []
    while not queue.empty():
        results.append(queue.get())
    
    return results

if __name__ == '__main__':

    mp.set_start_method(START_METHOD)

    # Sequential execution
    results_seq, sequential_time = sequential_execution(NUM_TASKS, TASK_COMPLEXITY)
    print(f"Sequential execution: {sequential_time:.2f} seconds")

    # Parallel execution
    results_par, parallel_time = parallel_execution(NUM_TASKS, TASK_COMPLEXITY)
    
    print(f"Parallel execution: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    assert results_seq == results_par
    print("OK")
    print("results: ", results_par)
    
    
