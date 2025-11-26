import os, sys, sysconfig
import warnings, importlib, sys, sysconfig
import time
import json


TASK_COMPLEXITY = 10000000
NUM_TASKS = 10
START_METHOD = 'spawn'  # 'fork' , 'spawn' or 'forkserver'
NUM_WORKERS = os.cpu_count()
NUM_REPS = 2 # number of repetitions for averaging timings
RESULTS_FILE = "results/times.json"

def _ensure_no_gil():
    supports_ft = sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    is_gil_on = getattr(sys, "_is_gil_enabled", lambda: None)
    is_gil_on = (is_gil_on() if is_gil_on is not None else None)

    # If this build doesn't support free-threading, bail early.
    if not supports_ft:
        raise RuntimeError(
            "This interpreter is not a free-threaded build. "
            "Install/run a free-threaded (no-GIL) CPython and try again."
        )

    # If GIL is enabled, relaunch with it disabled.
    if is_gil_on is True:
        os.environ["PYTHON_GIL"] = "0"
        # prepend -X gil=0 so it takes precedence
        args = [sys.executable, "-X", "gil=0", *sys.argv]
        os.execv(sys.executable, args)

def check_libs_no_gil(listlibs=["numpy", "pandas", "scikit-learn"]):
    """
    Helper function to import libraries and check if they trigger GIL warnings
    or re-enable the GIL.
    """
    def import_and_check(name):
        was_gil = getattr(sys, "_is_gil_enabled", lambda: None)()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            m = importlib.import_module(name)
        now_gil = getattr(sys, "_is_gil_enabled", lambda: None)()
        for w in rec:
            if isinstance(w.message, RuntimeWarning) and "GIL" in str(w.message):
                print(f"[WARN] {name} triggered: {w.message}")
        if was_gil is False and now_gil is True:
            print(f"[ALERT] Importing {name} re-enabled the GIL")
        return m
    for mod in listlibs:
        try:
            import_and_check(mod)
        except Exception as e:
            print(f"[ERROR] importing {mod}: {e}")

def measure_time_decorator(times=NUM_REPS):
    """
    Decorator to measure execution time of a function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Measuring time for {func.__name__} over {times} runs...")
            avg_time = 0
            for _ in range(times):
                print(f"Run {_+1} of {times}...")
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                execution_time = end - start
                avg_time += execution_time
            avg_time /= times
            return result, avg_time
        return wrapper
    return decorator

def cpu_intensive_task(n = TASK_COMPLEXITY):
    """Simulate a CPU-intensive task"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

@measure_time_decorator(times=NUM_REPS)
def sequential_execution(num_tasks=NUM_TASKS, task_complexity=TASK_COMPLEXITY):
    results = [cpu_intensive_task(task_complexity) for _ in range(num_tasks)]
    return results

def store_results(file_name, name_method, sequential_time, parallel_time, speedup, num_reps=NUM_REPS):
    """
    Store average execution time and number of repetitions in a JSON file.

    name_method [str]: Name of the parallelization method used (e.g., "Threads", "Process", etc.)
    file_name [str]: Path to the JSON file where results are stored
    """
    try:
        full_json = json.loads(open(file_name).read())
    except Exception as e:
        print(f"[ERROR] {e}")

        full_json = {}

    prev_times = full_json.get(name_method, {})
    prev_avg_parallel_time = prev_times.get("avg_parallel_time", None)
    prev_avg_sequential_time = prev_times.get("avg_sequential_time", None)
    prev_avg_speedup = prev_times.get("avg_speedup", None)
    prev_num_reps = prev_times.get("num_reps", 0)

    new_avg_parallel_time = (prev_avg_parallel_time+parallel_time)/2 if prev_avg_parallel_time is not None else parallel_time
    new_avg_sequential_time = (prev_avg_sequential_time+sequential_time)/2 if prev_avg_sequential_time is not None else sequential_time
    new_avg_speedup = (prev_avg_speedup+speedup)/2 if prev_avg_speedup is not None else speedup
    new_num_reps = prev_num_reps + num_reps
    prev_times = {
        "avg_sequential_time": new_avg_sequential_time,
        "avg_parallel_time": new_avg_parallel_time,
        "avg_speedup": new_avg_speedup,
        "num_reps": new_num_reps
    }
    full_json[name_method] = prev_times


    with open(file_name, 'w') as f:
        json.dump(full_json, f, indent=4)