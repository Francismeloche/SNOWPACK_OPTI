import os
import time
import shutil
import itertools
import numpy as np
import pandas as pd
import concurrent.futures
import sys

def stringify(iteration):
    base = list('00000')
    str_iteration = str(iteration)
    for idx, letter in enumerate(str_iteration):
        base[-(len(str_iteration) - idx)] = letter
        
    return(''.join(base)) 


parameters = {
    'WIND_SCALING_FACTOR': [1],#list(np.arange(1, 3.1, 0.1).round(2)),
    'TRESH_RAIN': [1.2], #list(np.arange(-0.5, 2.4, 0.1).round(2)),
    'ATMOSPHERIC_STABILITY': ['NEUTRAL','RICHARDSON', 'MO_LOG_LINEAR', 'MO_HOLTSLAG', 'MO_STEARNS', 'MO_MICHLMAYR','MO_SCHLOEGL_UNI','MO_SCHLOEGL_MULTI','MO_SCHLOEGL_MULTI_OFFSET'],
    #'ATMOSPHERIC_STABILITY': ['NEUTRAL','MO_MICHLMAYR','MO_SCHLOEGL_UNI','MO_SCHLOEGL_MULTI','MO_SCHLOEGL_MULTI_OFFSET'],
    #'HN_DENSITY_PARAMETERIZATION': ['LEHNING_NEW','BELLAIRE','ZWART','PAHAUT','NIED','VANKAMPENHOUT'],
    'HN_DENSITY_PARAMETERIZATION': ['BELLAIRE', 'LEHNING_NEW'],
    'ALBEDO_PARAMETERIZATION': ['LEHNING_0','LEHNING_1', 'LEHNING_2','SCHMUCKI_GSZ','SCHMUCKI_OGS'],
    #'ALBEDO_PARAMETERIZATION': ['LEHNING_2'],
    'HARDNESS_PARAMETERIZATION': ['MONTI', 'BELLAIRE', 'ASARC'],
    'STATION1': ['INSITU','DWNSKL']
    
}
names = sorted(parameters)
combinations = list(itertools.product(*[parameters[name] for name in names]))
wf_parameters = [c + (stringify(idx),) for idx, c in enumerate(combinations)]


#define the path to the base model files (ini and IO)
MODEL_PATH = sys.argv[1]
#define the path where to write the optimization simulation files
PARENT_DIR = sys.argv[2]

def write_file(parameters):
    # Unpack specific run parameters
    albedo_param, atm_model, hard_param, HN_density_param, station, thresh_rain, wind_scaling, iteration = parameters
    # Generate run id
    run_id = f'FID_{stringify(iteration)}'
    # Create destination path
    dir_path = os.path.join(PARENT_DIR, run_id)
    # Create new directory and copy model run directories
    shutil.copytree(MODEL_PATH, dir_path)
    # Rename ini file
    prev_name = os.path.join(dir_path, 'cfgfiles/ini_model')
    #print(prev_name)
    new_name = os.path.join(dir_path, f'cfgfiles/{run_id}')
    #print(new_name)
    os.rename(prev_name, new_name)
    # Modify ini file parameters for the run
    if station == 'INSITU':
        station_path = 'FidelityPatched_15min_2019.smet'
    elif station == 'DWNSKL':
        station_path = '30_55_gnp_meteo15min.smet'
    with open(new_name, 'r') as model:
        lines = model.readlines()
    lines[9] = f'METEOPATH = {dir_path}/input\n'
    lines[11] = f'STATION1 = {station_path}\n'
    lines[20] = f'METEOPATH = {dir_path}/output\n'
    lines[31] = f'SNOWPATH = {dir_path}/output\n'
    lines[84] = f'THRESH_RAIN = {thresh_rain}\n'
    lines[81] = f'WIND_SCALING_FACTOR = {wind_scaling}\n'
    lines[65] = f'ATMOSPHERIC_STABILITY = {atm_model}\n'
    lines[101] = f'HN_DENSITY_PARAMETERIZATION = {HN_density_param}\n'
    lines[123] = f'ALBEDO_PARAMETERIZATION = {albedo_param}\n'
    lines[109] = f'HARDNESS_PARAMETERIZATION = {hard_param}\n'
    if atm_model != 'NEUTRAL':
        lines[68] = 'CHANGE_BC = FALSE\n'
        # Delete TRESH_CHANGE_BC line that is not needed except for Neutral atmospheric model
        del lines[69]
    # Write changes to file
    with open(new_name, 'w') as f:
        f.writelines(lines)
    # Modifiy .sno file
    with open(f'{dir_path}/input/30_55_gnp.sno', 'r') as sno_model:
        lines = sno_model.readLines()
    lines[20] = f'WindScalingFactor = {wind_scaling}\n'
    with open(f'{dir_path}/input/30_55_gnp.sno', 'w') as sno_model:
        sno_model.writeLines(lines)

params_table = pd.DataFrame(wf_parameters, columns=['ALBEDO_PARAMETERIZATION','ATMOSPHERIC_MODEL','HARDNESS_PARAMETERIZATION','HN_DENSITY_PARAMETERIZATION','STATION','TRESH_RAIN', 'WIND_SCALING_FACTOR', 'ID'])
params_table.to_csv('./FID_OPTIMIZATION/params_table.csv')
# execute the wiritingfunction in parallel
t1 = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(write_file, wf_parameters)
    
t2 = time.perf_counter()

print(f'Writing simulation files finished in {t2 - t1}')
