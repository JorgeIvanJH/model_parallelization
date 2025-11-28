import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import joblib

from src.utils import measure_time_decorator
import numpy as np

NUM_ROWS = int(1e8) # Number of rows to read from the dataset



ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
DATASET_PATH = os.path.join(ROOT_DIR, "data", "healthcare_noshows_appointments.csv")
MODEL_PATH = os.path.join(ROOT_DIR,"src","ml","saved_models", "LogisticRegression_1150_28112025.joblib")
NUM_PROCESSES = os.cpu_count()

def load_dataset(DATASET_PATH=DATASET_PATH, NUM_ROWS=NUM_ROWS):
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

@measure_time_decorator(times=5)
def sequential_execution(model, X, y):
    y_hat = model.predict(X)
    return y_hat

@measure_time_decorator(times=5)
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

if __name__ == "__main__":

    X, y = load_dataset()
    model = joblib.load(MODEL_PATH)

    print("TESTING ON DATAWET WITH {} ROWS".format(X.shape[0]))

    # Sequential execution
    y_hat_seq, seq_time = sequential_execution(model, X, y)
    print(f"Sequential execution time: {seq_time:.2f} seconds")
    # Parallel execution
    y_hat_par, par_time = parallel_execution(model, X, y, n_jobs=NUM_PROCESSES)
    print(f"Parallel execution time: {par_time:.2f} seconds")

    # Verify results are the same
    assert np.array_equal(y_hat_seq, y_hat_par), "Predictions from sequential and parallel execution do not match!"

    # Speedup
    speedup = seq_time / par_time
    print(f"Speedup: {speedup:.2f}x")
    print("OK")
