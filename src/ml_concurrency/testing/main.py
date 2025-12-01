import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import joblib
import numpy as np
from ml_concurrency.utils import load_dataset, sequential_execution, parallel_execution, store_results
import time


NUM_ROWS = int(1e6) # Number of rows to read from the dataset
NUM_PROCESSES = os.cpu_count()
NUM_REPS = 1
MODEL_NAME = "LogisticRegression.joblib"


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "..",".."))
DATASET_PATH = os.path.join(ROOT_DIR, "data", "healthcare_noshows_appointments.csv")
MODEL_PATH = os.path.join(ROOT_DIR,"src","ml_concurrency","testing","ml_models", MODEL_NAME)
RESULTS_PATH = os.path.join(ROOT_DIR, "src", "ml_concurrency", "testing", "results", "runs.csv")



if __name__ == "__main__":

    X, y = load_dataset(DATASET_PATH, NUM_ROWS)
    model = joblib.load(MODEL_PATH)

    print("TESTING ON DATAWET WITH {} ROWS".format(X.shape[0]))

    # Sequential execution
    y_hat_seq, sequential_time = sequential_execution(model, X, y)
    print(f"Sequential execution time: {sequential_time:.2f} seconds")
    # Parallel execution
    y_hat_par, parallel_time = parallel_execution(model, X, y, n_jobs=NUM_PROCESSES)
    print(f"Parallel execution time: {parallel_time:.2f} seconds")

    # Verify results are the same
    assert np.array_equal(y_hat_seq, y_hat_par), "Predictions from sequential and parallel execution do not match!"

    # Speedup
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

    # Store Results
    store_results(RESULTS_PATH, MODEL_NAME.split(".")[0], NUM_PROCESSES, NUM_ROWS, sequential_time, parallel_time)
    print("OK")
    