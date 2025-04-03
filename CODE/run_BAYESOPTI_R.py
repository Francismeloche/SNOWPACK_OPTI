import numpy as np
import subprocess
import os
import sys
import re
from subprocess import TimeoutExpired
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args, create_result
from skopt import Optimizer
from skopt.plots import plot_convergence, plot_objective
from skopt import dump, load

from joblib import Parallel, delayed


np.int = int  # Monkey patch deprecated np.int

ROOT = os.path.abspath("./MODEL_15minBAYES/")

# Define the search space for SNOWPACK parameters
space = [
    Real(-1, 3, name='thresh_rain'),
    Real(-1, 3, name='hoar_thresh_TA'),
    Real(0.6, 1, name='hoar_thresh_RH'),
    Real(1, 10, name='hoar_thresh_VW'),
    Integer(1,3, name = 'wind_scaling'),
    Categorical(['NEUTRAL', 'MO_LOG_LINEAR','MO_SCHLOEGL_MULTI','MO_SCHLOEGL_MULTI_OFFSET'], name = 'atmo_model'),
    Categorical(['LEHNING_0','LEHNING_1', 'LEHNING_2','SCHMUCKI_GSZ','SCHMUCKI_OGS'], name = 'albedo_model'),
    Categorical(['LEHNING_NEW','BELLAIRE','ZWART','PAHAUT','NIED','VANKAMPENHOUT'], name = 'HN_density_model')
]

import shutil
import tempfile
import uuid

def run_snowpack(parameters):
    # Unpack parameters
    thresh_rain, hoar_thresh_TA, hoar_thresh_RH, hoar_thresh_VW, wind_scaling, atmo_model, albedo_model, HN_density_model = parameters

    # Create a unique temporary directory for this run in PARALLEL RUNNING
    temp_dir = os.path.join(ROOT, "temp_run", f"run_{uuid.uuid4().hex}")  # Unique per process
    os.makedirs(temp_dir, exist_ok=True)

    # Copy necessary files to temp directory
    shutil.copytree(os.path.join(ROOT, 'cfgfiles'), os.path.join(temp_dir, 'cfgfiles'))
    shutil.copytree(os.path.join(ROOT, 'input'), os.path.join(temp_dir, 'input'))
    shutil.copytree(os.path.join(ROOT, 'output'), os.path.join(temp_dir, 'output'))

    # Modify the copied ini file inside the temp directory
    ini_file = os.path.join(temp_dir, 'cfgfiles/ini_model')

    with open(ini_file, 'r') as model:
        lines = model.readlines()

    # Modify parameters
    lines[9] = f'METEOPATH = {temp_dir}/input\n'
    lines[11] = f'STATION1 = FidelityPatched_15min_2019.smet\n'
    lines[20] = f'METEOPATH = {temp_dir}/output\n'
    lines[31] = f'SNOWPATH = {temp_dir}/output\n'
    lines[65] = f'ATMOSPHERIC_STABILITY = {atmo_model}\n'
    lines[84] = f'THRESH_RAIN = {thresh_rain}\n'
    lines[81] = f'WIND_SCALING_FACTOR = {wind_scaling}\n'
    lines[86] = f'HOAR_THRESH_TA = {hoar_thresh_TA}\n'
    lines[87] = f'HOAR_THRESH_RH = {hoar_thresh_RH}\n'
    lines[88] = f'HOAR_THRESH_VW = {hoar_thresh_VW}\n'
    lines[101] = f'HN_DENSITY_PARAMETERIZATION = {HN_density_model}\n'
    lines[123] = f'ALBEDO_PARAMETERIZATION = {albedo_model}\n'

    # Save the modified ini file
    with open(ini_file, 'w') as f:
        f.writelines(lines)
    f.close()
    # Modify .sno file
    sno_file = os.path.join(temp_dir, 'input/30_55_gnp.sno')
    with open(sno_file, 'r') as sno_model:
        lines = sno_model.readlines()
    sno_model.close()
    #lines[30] = f'WindScalingFactor = {wind_scaling}\n'

    with open(sno_file, 'w') as sno_model:
        sno_model.writelines(lines)
    sno_model.close()
    try:
        # Run Snowpack in the temporary directory
        snowpack_process = subprocess.Popen(
            ["snowpack", "-c", f"{temp_dir}/cfgfiles/ini_model", "-e", "2019-06-04T13:00:00"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        snow_stdout, snow_stderr = snowpack_process.communicate()  # 5 min timeout
        if snowpack_process.returncode != 0:
            print(f"Snowpack error: {snow_stderr}")

        # Run post-processing Python script
        python_script_path = os.path.abspath("./CODE/Similarity_surfBAYES.py")
        rscript_path = os.path.abspath("./CODE/SimilarityBAYES.R")

        python_process = subprocess.Popen(
            [sys.executable, python_script_path, temp_dir],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Run the R script
        rscript_process = subprocess.Popen(
            ["Rscript", rscript_path, temp_dir],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        r_stdout, r_stderr = rscript_process.communicate()
        py_stdout, py_stderr = python_process.communicate()
        # Extract RMSE from stdout using regex
        match = re.search(r"F1:\s*([\d\.]+)", py_stdout)
        match2 = re.search(r"RMSE_HS:\s*([\d\.]+)", py_stdout)
        if match:
            F1 = float(match.group(1))
            rmse = float(match2.group(1))
        else:
            print("F1 not found in SNOWPACK output!")
            return float(1)

        print(f"SNOWPACK RMSE HS: {rmse}")
        print(f"SNOWPACK F1: {F1}")

         # Extract similarity from stdout using regex
        match = re.search(r"simi:\s*([\d\.]+)", r_stdout)
        if match:
            simi = float(match.group(1))
        else:
            print("simi not found in SNOWPACK output!")
            return float(1)

        print(f"SNOWPACK simi: {simi}")
        print('FINAL score :', (F1*0.4 + simi*0.3 + rmse*0.3))
        return float(F1*0.4 + simi*0.3 + rmse*0.3)
        #print('FINAL score :', simi)
        #return float(simi)

    except TimeoutExpired:
        print("Snowpack process timed out!")
        snowpack_process.kill()
        return float("inf")

    except Exception as e:
        print(f"Error running SNOWPACK: {e}")
        return float("inf")

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Failed to delete temp directory: {e}")



# Number of parallel evaluations per iteration
n_parallel = 24
n_iterations = 5 # Total Bayesian optimization iterations

# Bayesian optimizer using Gaussian Process
optimizer = Optimizer(space, base_estimator="GP", acq_func="EI", random_state=2)

# Run Bayesian Optimization in Parallel
for i in range(n_iterations):
    print(f"Iteration {i+1}/{n_iterations}")

    # Step 1: Ask for `n_parallel` new candidate points
    x_batch = optimizer.ask(n_points=n_parallel)

    # Step 2: Evaluate all candidate points in parallel
    y_batch = Parallel(n_jobs=n_parallel)(delayed(run_snowpack)(x) for x in x_batch)

    # Step 3: Tell the optimizer the results
    optimizer.tell(x_batch, y_batch)

# Get best found parameters
best_paramsMIN = optimizer.Xi[np.argmin(optimizer.yi)]
best_scoreMIN = min(optimizer.yi)


print("Best parameters:", best_paramsMIN)
print("Best score:", best_scoreMIN)

result = create_result(optimizer.Xi, optimizer.yi,
                             space = optimizer.space,
                             rng = optimizer.rng,
                             specs = optimizer.specs,
                             models = optimizer.models)
dump(result, './CODE/result_opti.pkl')

try:
    # --- PLOT CONVERGENCE ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_convergence(result, ax=ax1)
    ax1.set_title("Bayesian Optimization Convergence")
    plt.savefig("./CODE/convergence_plot.jpg", dpi=300,bbox_inches='tight')
    plt.close(fig1)

    # --- PLOT OBJECTIVE FUNCTION (if applicable) ---
    #if len(optimizer.Xi[0]) <= 2:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_objective(result, sample_source ='result', n_points =5)
    ax2.set_title("Objective Function Landscape")
    plt.savefig("./CODE/objective_plot.jpg", dpi=300,bbox_inches='tight')
    plt.close(fig2)

    print("Plots saved: convergence_plot.jpg and (if applicable) objective_plot.jpg")
except Exception as e:
    print(f"Error creating plots: {e}")

