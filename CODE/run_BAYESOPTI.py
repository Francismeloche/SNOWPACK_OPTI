import numpy as np
import subprocess
import os
import sys
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from joblib import Parallel, delayed
import re
from subprocess import TimeoutExpired

np.int = int  # Monkey patch deprecated np.int

ROOT = os.path.abspath("./MODEL_15minBAYES/")

# Define the search space for SNOWPACK parameters
space = [
    Integer(1, 3, name='wind_scaling_factor'),
    Real(-1, 2, name='thresh_rain'),
    Real(-1, 2, name='hoar_thresh_TA'),
    Real(0.8, 1, name='hoar_thresh_RH'),
    Real(1, 10, name='hoar_thresh_VW')
]

# Function to run SNOWPACK and evaluate performance
def run_snowpack(parameters):
    # Unpack specific run parameters
    wind_scaling, thresh_rain, hoar_thresh_TA, hoar_thresh_RH, hoar_thresh_VW = parameters

    # Rename ini file
    ini_file = os.path.join(ROOT, 'cfgfiles/ini_model')
    dir_path = ROOT

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
            ["snowpack", "-c", f"{ROOT}/cfgfiles/ini_model", "-e", "2019-06-04T13:00:00"],
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
            return 1

        print(f"SNOWPACK F1: {F1}")
        return F1

    except TimeoutExpired:
        print("Snowpack process timed out!")
        snowpack_process.kill()
        return float("inf")

    except Exception as e:
        print(f"Error running SNOWPACK: {e}")
        return float("inf")


# Wrapper for Bayesian optimization
@use_named_args(space)
def objective(**parameters):
    return run_snowpack([parameters['wind_scaling_factor'], parameters['thresh_rain'], 
                         parameters['hoar_thresh_TA'], parameters['hoar_thresh_RH'], 
                         parameters['hoar_thresh_VW']])

# Perform Bayesian optimization
result = gp_minimize(
    objective, space,
    n_calls=100,  # Total iterations
    n_initial_points=5,  # Random initial points
    acq_func="EI",  # Expected improvement strategy
    n_jobs=-1,  # Parallel function evaluations handled within the objective function
    random_state=42
)

# Print best parameters
print("Best parameters:", result.x)
print("Best F1:", 1-result.fun)
from skopt.plots import plot_convergence
plot_convergence(results)

