#!/bin/bash

# IMPORTANT FOR CC STACK COMPUTATION

#temporary create environement path insteat of the /bashrc file
#Meteoio
export PATH="/home/fmeloche/meteoio/bin/bin:$PATH"
export LD_LIBRARY_PATH="/home/fmeloche/meteoio/bin/lib:$LD_LIBRARY_PATH"

#Snowpack
export PATH="/home/fmeloche/snowpack/bin/bin:$PATH"
export LD_LIBRARY_PATH="/home/fmeloche/snowpack/bin/lib:$LD_LIBRARY_PATH"

module load python/3.11 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env_opti
source $SLURM_TMPDIR/env_opti/bin/activate
pip install --no-index --upgrade pip
pip install --no-index scikit-learn scikit-optimize joblib

module load r/4.4.0