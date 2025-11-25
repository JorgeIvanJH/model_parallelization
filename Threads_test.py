import threading
import queue
import time

from utils import cpu_intensive_task, sequential_execution, measure_time_decorator, _ensure_no_gil
from utils import TASK_COMPLEXITY, NUM_TASKS

def worker(result_queue, task_complexity):
    result = cpu_intensive_task(task_complexity)
    result_queue.put(result)

@measure_time_decorator
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

    # Parallel execution (threaded)
    results_par, parallel_time = parallel_execution(NUM_TASKS, TASK_COMPLEXITY)
    
    print(f"Parallel execution (threads): {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    assert results_seq == results_par
    print("OK")
    print("results:", results_par)
