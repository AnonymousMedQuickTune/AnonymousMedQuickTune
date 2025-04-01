#!/bin/bash
# Exit on first error
set -e

# Run pylint on src folder, ignoring specific warnings
python -m pylint --disable=R0912,R0913,R0914,R0915,R0917,W0718,no-member src/*

