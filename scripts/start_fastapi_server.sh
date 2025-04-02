#!/bin/bash

# Prerequisites
# sudo apt update && sudo apt-get install -y build-essential python3-dev python3-venv libgl1 libglib2.0-0

VENV_DIR=".venv"
set -x

# GEt location of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#move one directory up
cd $DIR
cd ..

# Try to install the virtual environment if it does not exist
if [ ! -d $VENV_DIR ]; then
    python3 -m venv $VENV_DIR
fi

if [ "$(uname)" == "Darwin" ] || [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    source $VENV_DIR/bin/activate
else
    source $VENV_DIR/Scripts/activate
fi

pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

#Find the correct Python directory and apply the change to basicsr/data/degradations.py
#This is a known issue, here we are just automating the process
# See the issue here: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-1813885266
PYTHON_SITE_PACKAGES=$(find "$VENV_DIR/lib" -name "site-packages" | head -n 1)
DEGRADATIONS_PY="$PYTHON_SITE_PACKAGES/basicsr/data/degradations.py"

if [ -f "$DEGRADATIONS_PY" ]; then
  echo "Applying changes to $DEGRADATIONS_PY..."
  sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEGRADATIONS_PY"
else
  echo "File $DEGRADATIONS_PY not found. Skipping modification."
fi
# Change by Fukuda 2024/06/03

cd server
#python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
CORES=$(nproc)
WORKERS=$CORES # When the instance is running on an AWS instance which works on vCPUs
THREADS=$(( 3 * CORES))
if [ "$WORKERS" -ge 10 ]; then
    WORKERS=10
    THREADS=$((3 * WORKERS))
fi
if [ "$ENV_MODE" = "production" ] || [ "$ENV_MODE" = "staging" ]; then
  rm -rf .env
  echo "Starting Gunicorn in production mode with $WORKERS workers......"
  gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8002 --log-config app/log_conf.conf --timeout 600 --threads $THREADS --max-requests 5 --max-requests-jitter 5 --graceful-timeout 600
elif [ "$ENV_MODE" = "test" ]; then
  echo "$ENV_MODE is set to test or an unknown mode. Not starting Gunicorn."
  pip3 install pytest && pytest --junitxml=junit.xml || ((($? == 5)) && echo 'Did not find any tests to run.')
else
  echo "Starting Gunicorn in development mode with $WORKERS workers......"
  gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8002 --timeout 600 --threads $THREADS --max-requests 5 --max-requests-jitter 5 --graceful-timeout 600
fi
