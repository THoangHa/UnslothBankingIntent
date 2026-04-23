#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e 

echo "========================================"
echo "  Commencing Intent Classification Training  "
echo "========================================"

# Run the training Python script
python scripts/train.py

echo "========================================"
echo "        Training Phase Completed        "
echo "========================================"