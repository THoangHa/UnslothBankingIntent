#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e 

echo "========================================"
echo " Initialising Standalone Inference Test "
echo "========================================"

# Run the inference Python script
python scripts/inference.py

echo "========================================"
echo "          Inference Completed           "
echo "========================================"