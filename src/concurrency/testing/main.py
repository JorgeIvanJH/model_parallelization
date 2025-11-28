import subprocess
from pathlib import Path

# Paths to Python executables
python314_NOGIL = r"C:\Users\jjaramil\AppData\Local\anaconda3\envs\python314nogil\python.exe"
python314       = r"C:\Users\jjaramil\AppData\Local\anaconda3\envs\python314\python.exe"

# Test pairs: (python, script)
TESTS = [
    (python314_NOGIL, r"concurrency_methods\Threads_test.py"),
    (python314,       r"concurrency_methods\Process_test.py"),
    (python314,       r"concurrency_methods\Pool_test.py"),
    (python314,       r"concurrency_methods\ProcessPoolExecutor_test.py"),
]

def run_script(python_path, script_path):
    subprocess.run(
        [python_path, script_path],
        capture_output=False,
        text=True,
        check=True
    )

if __name__ == "__main__":

    for python_path, script_path in TESTS:
        print(f"Running {script_path} with {python_path}...")
        run_script(python_path, script_path)
