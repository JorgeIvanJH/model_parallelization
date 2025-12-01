import threading
import queue
from src.utils import cpu_intensive_task, sequential_execution, measure_time_decorator, _ensure_no_gil, store_results
from src.utils import TASK_COMPLEXITY, NUM_TASKS, NUM_REPS, NUM_REPS, RESULTS_FILE

def worker(result_queue, task_complexity):
    result = cpu_intensive_task(task_complexity)
    result_queue.put(result)

@measure_time_decorator(times=NUM_REPS)
def parallel_execution(num_tasks, task_complexity):
    result_queue = queue.Queue()
    threads = []

    for i in range(num_tasks):
        t = threading.Thread(target=worker, args=(result_queue, task_complexity))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    return results

if __name__ == '__main__':

    _ensure_no_gil()

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
    store_results(RESULTS_FILE, "Threads",sequential_time, parallel_time, speedup, NUM_REPS)


