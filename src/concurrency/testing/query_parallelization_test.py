import os
import multiprocessing as mp
from collections import Counter
import time
import numpy as np
from functools import wraps
TIMEPERDAY = 1.0  # seconds to simulate query time per day
NUMWORKERS = os.cpu_count()  # number of parallel workers to use

def measure_time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper

def split_into_workers(lst, num_workers):
    """
    divides lst into num_workers sublists of (approximately) equal size
    """
    k, m = divmod(len(lst), num_workers)
    return [lst[i*k + min(i, m) : (i+1)*k + min(i+1, m)] for i in range(num_workers)]

def query(day, query_time=TIMEPERDAY):
    """
    Simulate query of data for a given day
    """
    time.sleep(query_time)
    return "batch_day_" + str(day)

def batch_query(days):
    """
    simulate batch query of data for multiple days
    """
    numdays = len(days)
    query_time = TIMEPERDAY + np.sqrt(numdays)
    time.sleep(query_time) # TODO: Adjust to a more realistic approach
    return ["batch_day_" + str(day) for day in days]

def model(data,inference_time=0.1):
    """
    Simulate inference on a model
    """
    time.sleep(inference_time)
    return f"processed_{data}"

def worker(raw_data_queue, day):
    raw_data = query(day)
    raw_data_queue.put(raw_data)

def batch_worker(raw_data_queue, days_batch):
    raw_data = batch_query(days_batch)
    for data in raw_data:
        raw_data_queue.put(data)

def worker_full(day):
    """Do both query and model in the same worker process."""
    raw = query(day)
    return model(raw)

@measure_time_decorator
def sequential_approach(days):
    """
    query and model are both called sequentially
    """
    processed_data = []
    for day in days:
        raw_data = query(day)
        processed = model(raw_data)
        processed_data.append(processed)
    return processed_data

@measure_time_decorator
def single_query_approach(days):
    """
    query for all days in a single call
    model is called once for all data
    """
    raw_data = batch_query(days)
    processed_data = [model(data) for data in raw_data]
    return processed_data

@measure_time_decorator
def approach_1(days):
    """
    query is called in parallel, 1 process per day
    model is called sequentially
    """
    raw_data_queue = mp.Queue()
    processes = []
    for day in days:
        p = mp.Process(target=worker, args=(raw_data_queue, day))
        processes.append(p)
        p.start()
    for p in processes: # Wait for all processes to complete
        p.join()
    processed_data = []
    while not raw_data_queue.empty():
        raw_data = raw_data_queue.get()
        processed = model(raw_data)
        processed_data.append(processed)
    return processed_data

@measure_time_decorator
def approach_2(days,num_workers=NUMWORKERS):
    """
    query is called in parallel batches, [num_workers] processes
    model is called sequentially
    """
    days_batches = split_into_workers(days, num_workers)
    raw_data_queue = mp.Queue()
    processes = []
    for days_batch in days_batches:
        p = mp.Process(target=batch_worker, args=(raw_data_queue, days_batch))
        processes.append(p)
        p.start()
    for p in processes: # Wait for all processes to complete
        p.join()
    processed_data = []
    while not raw_data_queue.empty():
        raw_data = raw_data_queue.get()
        processed = model(raw_data)
        processed_data.append(processed)
    return processed_data

@measure_time_decorator
def approach_3(days, num_workers=NUMWORKERS):
    """
    query is called in parallel batches, [num_workers] processes
    model is called asynchronously right after each query result is available
    """
    days_batches = split_into_workers(days, num_workers)
    raw_data_queue = mp.Queue()
    processes = []
    for days_batch in days_batches:
        p = mp.Process(target=batch_worker, args=(raw_data_queue, days_batch))
        processes.append(p)
        p.start()

    processed_data = []

    num_results = len(days)
    for _ in range(num_results):
        raw_data_val = raw_data_queue.get()  # blocks until next item is available
        processed = model(raw_data_val)
        processed_data.append(processed)

    for p in processes: # Wait for all processes to complete
        p.join()

    return processed_data

@measure_time_decorator
def approach_4(days, num_workers=NUMWORKERS):
    """
    query and model are both called in parallel using multiprocessing Pool
    """
    with mp.Pool(processes=num_workers) as pool:
        processed_data = pool.map(worker_full, days)
    return processed_data

if __name__ == '__main__':
    print(f"Testing with {NUMWORKERS} workers")

    days = list(range(1, 30))

    methods = [sequential_approach,single_query_approach, approach_1, approach_2, approach_3, approach_4]

    previous_processed_data = None
    for method in methods:
        processed_data, elapsed_time = method(days)
        print(f"method: {method.__name__}, time: {elapsed_time:.2f} seconds")
        # VERIFY CONSISTENCY ACROSS METHODS
        if previous_processed_data is not None:
            assert Counter(previous_processed_data) == Counter(processed_data), "Inconsistent results between methods!"
        previous_processed_data = processed_data

