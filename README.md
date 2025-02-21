## DISCLAIMER : ONGOING PROJECT

# SNOWPACK OPTIMIZATION
This is a a collection of script optimize the SNOWPACK parameters and parameterization from snow profile observations and snow crytals observed at the surface.
This was done at Fidelity weather station in Rogers Pass Canada with weather observations, and also NWP downscale virtual station. The code is written to be run on HPC of Compute Canada for approximately 500 simulations in 20min on 24 CPU and 32G memory.

1. The first code is ini_generation, which generated ini files for each simulation based on the tested parameters. The script takes the default simulation in MODEL_15MIN and modify it to add new parameters. This script will generate a new directory will all the simulations directory (config,input,output) with a assigned labels.

2. The main script to be run is run_v4_INIGEN_OPTI.py , which uses the several others script in CODE directory. The path are relative path with the primary working directory being SNOWPACK_OPTI.

3. The script Similarity_surf.py computes the snow grain type similariy between SNOWPACK and the daily manual obersations of snow surface grain types at Fidelity weather station. It also compute the the F1 score and confusion matrix of the prediction of surface hoar (SH) and precipitation particles (PP).

4. The R script Similarity assess the snow profile similarity of three snow profiles made at Fidelity weather station (camml files in profil_2018-2019 directory).
