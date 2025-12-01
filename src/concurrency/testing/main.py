import os, sys
import subprocess
from pathlib import Path

# Paths to Python executables
python314_NOGIL = r"C:\Users\jjaramil\AppData\Local\anaconda3\envs\python314nogil\python.exe"
python314       = r"c:\Users\jjaramil\AppData\Local\anaconda3\envs\python314\python.exe"

ROOT_DIR = Path(__name__).resolve().parent.parent.parent.parent
PATH_TO_TESTS = os.path.join(ROOT_DIR, "src", "concurrency", "methods")
# Test pairs: (python, script)
TESTS = [
    (python314_NOGIL, "src.concurrency.methods.Threads_test"),
    (python314,      "src.concurrency.methods.Process_test"),
    (python314,      "src.concurrency.methods.Pool_test"),
    (python314,      "src.concurrency.methods.ProcessPoolExecutor_test"),
]


def run_script(python_path, module_name):
    subprocess.run(
        [python_path, "-m", module_name],
        cwd=ROOT_DIR,
        capture_output=False,
        text=True,
        check=True,
    )

if __name__ == "__main__":
    for python_path, module_name in TESTS:
        print(f"Running {module_name} with {python_path}...")
        run_script(python_path, module_name)