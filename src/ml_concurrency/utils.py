import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tracemalloc
from functools import wraps
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import time
from datetime import datetime
import threading
import psutil


def load_dataset(DATASET_PATH, NUM_ROWS):
    """Load and preprocess the healthcare no-shows dataset"""

    df = pd.read_csv(DATASET_PATH, dtype={'PatientId': 'category',
                                        'AppointmentID': 'category',
                                        'Gender': 'category',
                                        'Neighbourhood': 'category',
                                        }, 
                                    parse_dates=['ScheduledDay', 
                                                'AppointmentDay'])
    df = df.drop(columns=['AppointmentID', 'PatientId'])
    n_rows = df.shape[0]
    nrepeats,remiainder = NUM_ROWS // n_rows , NUM_ROWS % n_rows
    df = pd.concat([df]*nrepeats + [df.sample(remiainder)], ignore_index=True)
    y = df["Showed_up"]
    X = df.drop(columns=["Showed_up"])
    return X, y


def measure_performance(include_children=False, sample_interval=0.01):
    """
    Decorator to measure execution time and peak RSS memory (in MB).

    - include_children=True will also include memory from child processes,
      which is what you want for parallel / ProcessPoolExecutor code.
    - sample_interval is how often (in seconds) memory is sampled.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            proc = psutil.Process(os.getpid())
            stop_event = threading.Event()
            peak_bytes = 0

            def memory_sampler():
                nonlocal peak_bytes
                while not stop_event.is_set():
                    try:
                        rss = proc.memory_info().rss
                        if include_children:
                            for child in proc.children(recursive=True):
                                try:
                                    rss += child.memory_info().rss
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        if rss > peak_bytes:
                            peak_bytes = rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    time.sleep(sample_interval)

            # Start memory sampler thread
            sampler_thread = threading.Thread(target=memory_sampler)
            sampler_thread.daemon = True
            sampler_thread.start()

            # Time + function execution
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                end = time.perf_counter()
                # Stop sampler and wait for it to finish
                stop_event.set()
                sampler_thread.join()

            execution_time = end - start
            peak_mb = peak_bytes / (1024 ** 2)
            return result, execution_time, peak_mb

        return wrapper
    return decorator

@measure_performance(include_children=False)
def sequential_execution(model, X, y):
    y_hat = model.predict(X)
    return y_hat

@measure_performance(include_children=True)
def parallel_execution(model, X, y, n_jobs=-1):

    num_rows = X.shape[0]
    if num_rows == 0:
        return np.array([])

    # split the row indices so remainders are distributed evenly
    chunks = np.array_split(np.arange(num_rows), n_jobs)

    results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {executor.submit(model.predict, X.iloc[inds]): i for i, inds in enumerate(chunks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # concatenate in the original order and return (keeps alignment with X)
    return np.concatenate(results)

def store_results(file_dir, ml_model_name, num_workers, num_rows, sequential_time, parallel_time, sequential_memory, parallel_memory):
    """
    Read an existing CSV (if any) and append a new row with the provided information.
    Creates the file (and parent dirs) if it doesn't exist.
    """

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ml_model_name": ml_model_name,
        "num_workers": num_workers,
        "num_rows": num_rows,
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "sequential_memory": sequential_memory,
        "parallel_memory": parallel_memory
    }

    # read existing file or create new dataframe
    if os.path.exists(file_dir):
        try:
            df = pd.read_csv(file_dir)
        except Exception:
            df = pd.DataFrame(columns=list(row.keys()))
    else:
        df = pd.DataFrame(columns=list(row.keys()))

    # append and save
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(file_dir, index=False)
