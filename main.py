import subprocess
from pathlib import Path

# Paths to Python executables
python314_NOGIL = r"C:\Users\jjaramil\AppData\Local\anaconda3\envs\python314nogil\python.exe"
python314       = r"C:\Users\jjaramil\AppData\Local\anaconda3\envs\python314\python.exe"

# Test pairs: (python, script)
TESTS = [
    (python314_NOGIL, r"test_scripts\Threads_test.py"),
    (python314,       r"test_scripts\Process_test.py"),
    (python314,       r"test_scripts\Pool_test.py"),
    (python314,       r"test_scripts\ProcessPoolExecutor_test.py"),
]

def run_script(python_path, script_path):
    subprocess.run(
        [python_path, script_path],
        capture_output=True,
        text=True,
        check=True
    )

if __name__ == "__main__":

    for python_path, script_path in TESTS:
        print(f"Running {script_path} with {python_path}...")
        run_script(python_path, script_path)
