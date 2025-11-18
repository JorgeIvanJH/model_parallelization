import os
import time
TASK_COMPLEXITY = 10000000
NUM_TASKS = 10
NUM_WORKERS = os.cpu_count()

def measure_time_decorator(func):
    """
    Decorator to measure execution time of a function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        execution_time = end - start
        return result, execution_time
    return wrapper

def cpu_intensive_task(n = TASK_COMPLEXITY):
    """Simulate a CPU-intensive task"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

@measure_time_decorator
def sequential_execution(num_tasks=NUM_TASKS, task_complexity=TASK_COMPLEXITY):
    results = [cpu_intensive_task(task_complexity) for _ in range(num_tasks)]
    return results

