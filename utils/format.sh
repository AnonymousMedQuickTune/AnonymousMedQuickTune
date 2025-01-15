#!/bin/bash
set -e  # Exit on first failure

# Format code with black
python -m black src/*

# Sort imports with isort
python -m isort src/*