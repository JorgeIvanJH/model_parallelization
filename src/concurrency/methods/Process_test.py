import multiprocessing as mp
from src.concurrency.utils import cpu_intensive_task, sequential_execution, measure_time_decorator, store_results
from src.concurrency.utils import TASK_COMPLEXITY, NUM_TASKS, START_METHOD, NUM_REPS, RESULTS_FILE

def worker(queue, task_complexity):
    result = cpu_intensive_task(task_complexity)
    queue.put(result)

@measure_time_decorator(times=NUM_REPS)
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
    assert results_seq == results_par

    # Speedup
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")
    
    print("OK")
    
    # Store Results
    store_results(RESULTS_FILE, "Process",sequential_time, parallel_time, speedup, NUM_REPS)



