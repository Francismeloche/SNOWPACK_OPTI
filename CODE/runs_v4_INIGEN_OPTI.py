import os
import subprocess
import concurrent.futures
import time
import sys



#define the path to the base model files (ini and IO)
MODEL_PATH = './MODEL_15min/'

#define the path where to write the optimization simulation files
ROOT = "./FID_15/"

INIGEN_script_path = os.path.abspath("./CODE/ini_generation.py")
subprocess.run(
            [sys.executable, INIGEN_script_path, MODEL_PATH, ROOT],
             text=True
        )

dir_names = os.listdir(ROOT)

t1 = time.perf_counter()

def run_snowpack(dir_name):
    try:
        print(f"SNOWPACK {dir_name}")
        # Run Snowpack as a subprocess and enforce timeout
        snowpack_process = subprocess.Popen(
            ["snowpack", "-c", f"{ROOT}/{dir_name}/cfgfiles/{dir_name}", "-e", "2019-06-04T13:00:00"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for Snowpack to finish
        snow_stdout, snow_stderr = snowpack_process.communicate(timeout=300)  # 5 min timeout
        if snowpack_process.returncode != 0:
            print(f"Snowpack error in {dir_name}: {snow_stderr}")

        print(f"Snowpack Output for {dir_name}: {snow_stdout}")

        print(f"Post-Processing {dir_name}")
        # Ensure full paths
        rscript_path = os.path.abspath("./CODE/Similarity.R")
        python_script_path = os.path.abspath("./CODE/Similarity_surf.py")

        # Run the R script
        rscript_process = subprocess.Popen(
            ["Rscript", rscript_path, ROOT, dir_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Run the Python script
        python_process = subprocess.Popen(
            [sys.executable, python_script_path, ROOT, dir_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Capture output
        r_stdout, r_stderr = rscript_process.communicate(timeout=300)
        py_stdout, py_stderr = python_process.communicate(timeout=300)

        print(f"R Output for {dir_name}: {r_stdout}")
        print(f"Python Output for {dir_name}: {py_stdout}")

        if rscript_process.returncode != 0:
            print(f"Rscript error in {dir_name}: {r_stderr}")
        if python_process.returncode != 0:
            print(f"Python script error in {dir_name}: {py_stderr}")

    except subprocess.TimeoutExpired:
        print(f"Timeout expired for {dir_name}, terminating processes...")
        rscript_process.kill()
        python_process.kill()
    except Exception as e:
        print(f"Unexpected error in {dir_name}: {e}")


# Run Snowpack in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor: #adjust max workers for the number of CPU you have
    executor.map(run_snowpack, dir_names)

t2 = time.perf_counter()
print(f'Finished in {t2-t1} seconds')

# Force garbage collection to free memory
import gc
gc.collect()
