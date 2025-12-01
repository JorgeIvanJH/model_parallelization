import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import joblib
import numpy as np
from ml_concurrency.utils import load_dataset, sequential_execution, parallel_execution, store_results
from itertools import product
from tqdm import tqdm 

NUM_CORES = os.cpu_count()
MODEL_NAME = "LogisticRegression.joblib"


ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "..",".."))
DATASET_PATH = os.path.join(ROOT_DIR, "data", "healthcare_noshows_appointments.csv")
MODEL_PATH = os.path.join(ROOT_DIR,"src","ml_concurrency","testing","ml_models", MODEL_NAME)
RESULTS_PATH = os.path.join(ROOT_DIR, "src", "ml_concurrency", "testing", "results", "runs.csv")



if __name__ == "__main__":
    ROW_OPTIONS = [int(1e6), int(1e7), int(1e8)]
    PROCESS_OPTIONS = list(range(1, NUM_CORES + NUM_CORES // 2 , 3)) + [NUM_CORES * 2]

    comparison_pairs = list(product(ROW_OPTIONS, PROCESS_OPTIONS))

    for NUM_ROWS, NUM_PROCESSES in tqdm(
        comparison_pairs,
        desc="Benchmarking",
        unit="run"
    ):
        try:
            # Optional: show more info in the bar
            tqdm.write(f"\nRows={NUM_ROWS:,}, Procs={NUM_PROCESSES}")

            X, y = load_dataset(DATASET_PATH, NUM_ROWS)
            model = joblib.load(MODEL_PATH)

            tqdm.write(f"  Testing on dataset with {X.shape[0]:,} rows")

            y_hat_seq, sequential_time = sequential_execution(model, X, y)
            tqdm.write(f"  Sequential time: {sequential_time:.2f}s")

            y_hat_par, parallel_time = parallel_execution(
                model, X, y, n_jobs=NUM_PROCESSES
            )
            tqdm.write(f"  Parallel time:   {parallel_time:.2f}s")

            assert np.array_equal(y_hat_seq, y_hat_par), \
                "Predictions from sequential and parallel execution do not match!"

            speedup = sequential_time / parallel_time
            tqdm.write(f"  Speedup: {speedup:.2f}x")

            store_results(
                RESULTS_PATH,
                MODEL_NAME.split(".")[0],
                NUM_PROCESSES,
                NUM_ROWS,
                sequential_time,
                parallel_time,
            )
            tqdm.write("  Stored results")
        except Exception as e:
            tqdm.write(f"  Error: {e}")