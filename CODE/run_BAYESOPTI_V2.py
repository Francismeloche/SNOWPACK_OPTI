import numpy as np
import subprocess
import os
import sys
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from joblib import Parallel, delayed
import re
import concurrent.futures
from subprocess import TimeoutExpired

np.int = int  # Monkey patch deprecated np.int

MODEL_PATH = os.path.abspath("./MODEL_15minBAYES/")
OPTI_MODEL = os.path.abspath("./OPTIPATH_RUN/")

# Define the search space for SNOWPACK parameters
space = [
    Integer(1, 3, name='wind_scaling_factor'),
    Real(-1, 2, name='thresh_rain'),
    Real(-1, 2, name='hoar_thresh_TA'),
    Real(0.8, 1, name='hoar_thresh_RH'),
    Real(1, 10, name='hoar_thresh_VW')
]

# Function to run SNOWPACK and evaluate performance
def run_snowpack(seeds,parameters):
    # Unpack specific run parameters
    wind_scaling, thresh_rain, hoar_thresh_TA, hoar_thresh_RH, hoar_thresh_VW = parameters
    # Generate run id
    run_id = f'OPTIPATH_{seeds}'
    # Create destination path

    dir_path = os.path.join(OPTI_MODEL, run_id)
    # Create new directory and copy model run directories
    shutil.copytree(MODEL_PATH, dir_path)
    # Rename ini file
    prev_name = os.path.join(dir_path, 'cfgfiles/ini_model')
    #print(prev_name)
    new_name = os.path.join(dir_path, f'cfgfiles/{run_id}')
    #print(new_name)
    os.rename(prev_name, new_name)

    # Rename ini file
    ini_file = os.path.join(dir_path, f'/cfgfiles/{run_id}')


    # Modify ini file parameters for the run
    station_path = 'FidelityPatched_15min_2019.smet'
    with open(ini_file, 'r') as model:
        lines = model.readlines()
    lines[9] = f'METEOPATH = {dir_path}/input\n'
    lines[11] = f'STATION1 = {station_path}\n'
    lines[20] = f'METEOPATH = {dir_path}/output\n'
    lines[31] = f'SNOWPATH = {dir_path}/output\n'
    lines[84] = f'THRESH_RAIN = {thresh_rain}\n'
    lines[81] = f'WIND_SCALING_FACTOR = {wind_scaling}\n'
    lines[86] = f'HOAR_THRESH_TA = {hoar_thresh_TA}\n'
    lines[87] = f'HOAR_THRESH_RH = {hoar_thresh_RH}\n'
    lines[88] = f'HOAR_THRESH_VW = {hoar_thresh_VW}\n'

    # Write changes to file
    with open(ini_file, 'w') as f:
        f.writelines(lines)
    # Modify .sno file
    with open(f'{dir_path}/input/30_55_gnp.sno', 'r') as sno_model:
        lines = sno_model.readlines()
    lines[30] = f'WindScalingFactor = {wind_scaling}\n'
    with open(f'{dir_path}/input/30_55_gnp.sno', 'w') as sno_model:
        sno_model.writelines(lines)

    try:
        # Run Snowpack as a subprocess and enforce timeout
        snowpack_process = subprocess.Popen(
            ["snowpack", "-c", f"{dir_path}/cfgfiles/{run_id}", "-e", "2019-06-04T13:00:00"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for Snowpack to finish
        snow_stdout, snow_stderr = snowpack_process.communicate(timeout=300)  # 5 min timeout
        if snowpack_process.returncode != 0:
            print(f"Snowpack error: {snow_stderr}")

        # Post-processing and extracting output from another Python script
        python_script_path = os.path.abspath("./CODE/Similarity_surfBAYES.py")
        python_process = subprocess.Popen(
            [sys.executable, python_script_path, ROOT],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        py_stdout, py_stderr = python_process.communicate()
        # Extract RMSE from stdout using regex
        match = re.search(r"F1:\s*([\d\.]+)", py_stdout)
        if match:
            F1 = float(match.group(1))
        else:
            print("F1 not found in SNOWPACK output!")
            return float(1)

        print(f"SNOWPACK F1: {F1}")
        return float(F1)

    except TimeoutExpired:
        print("Snowpack process timed out!")
        snowpack_process.kill()
        return float("inf")

    except Exception as e:
        print(f"Error running SNOWPACK: {e}")
        return float("inf")


# Wrapper for Bayesian optimization
@use_named_args(space)
def objective(seeds,**parameters):
    return run_snowpack(seeds,list(parameters.values()))

n_parallel = 12  # Number of parallel runs

n_seeds = np.arange(1,13,1)

# Perform Bayesian optimization
def opti_bayes(seeds):
    result = gp_minimize(
        objective, space,
        n_calls=25,
        n_initial_points=10,
        acq_func="EI",
        random_state=seeds,
        n_jobs=-1 # Ensure parallel execution
        )
    return result

with concurrent.futures.ProcessPoolExecutor(max_workers=n_parallel) as executor: #adjust max workers for the number of CPU you have
    results = list(executor.map(opti_bayes, n_seeds))

# Print best parameters
for result in results:
    print("Best parameters:", results.x)
    print("Best F1:", 1-result.fun)
    from skopt.plots import plot_convergence
    plot_convergence(result)

