#!/bin/bash -l

# Deactivate virtualenv
deactivate

# Delete the virtual environment if it exists
if [ -d "myenv" ]; then
    rm -rf myenv
fi

# Create a new virtual environment
python3 -m pip install --user virtualenv
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

