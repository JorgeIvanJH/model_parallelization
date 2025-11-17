from multiprocessing import Process, Queue
import os
import time
from utils import cpu_intensive_task, sequential_execution
from utils import NUM_REPS, NUM_TASKS

def worker(queue, num_reps):
    """Worker function that puts result in queue"""
    result = cpu_intensive_task(num_reps)
    queue.put(result)
def parallel_execution(num_tasks, num_reps):
    start = time.time()
    queue = Queue()
    processes = []
    for i in range(num_tasks):
        p = Process(target=worker, args=(queue, num_reps))
        processes.append(p)
        p.start()
    # Wait for all processes to complete
    for p in processes:
        p.join()
    results = []
    while not queue.empty():
        results.append(queue.get())
    
    execution_time = time.time() - start
    return results, execution_time

if __name__ == '__main__':

    # Sequential execution
    results_seq, sequential_time = sequential_execution(NUM_TASKS, NUM_REPS)
    print(f"Sequential execution: {sequential_time:.2f} seconds")

    # Parallel execution
    results_par, parallel_time = parallel_execution(NUM_TASKS, NUM_REPS)
    print(f"Parallel execution: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    assert results_seq == results_par
    print("OK")
    print("results: ", results_par)
    
    
