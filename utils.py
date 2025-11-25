import os, sys, sysconfig
import warnings, importlib, sys, sysconfig
import time
TASK_COMPLEXITY = 10000000
NUM_TASKS = 10
START_METHOD = 'spawn'  # 'fork' , 'spawn' or 'forkserver'
NUM_WORKERS = os.cpu_count()

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

