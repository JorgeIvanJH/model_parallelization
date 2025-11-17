import os
import time
NUM_REPS = 10000000
NUM_TASKS = 40
NUM_WORKERS = os.cpu_count()

def cpu_intensive_task(n = NUM_REPS):
    """Simulate a CPU-intensive task"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

def sequential_execution(num_tasks=NUM_TASKS, num_reps=NUM_REPS):
    start = time.time()
    results = [cpu_intensive_task(num_reps) for _ in range(num_tasks)]
    end = time.time()
    execution_time = end - start
    return results, execution_time