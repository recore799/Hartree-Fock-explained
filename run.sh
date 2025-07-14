#!/bin/bash

# Clear numba cache to prevent issues
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name "__numba_cache__" -exec rm -rf {} +

# Run from repository root
PYTHONPATH="$(pwd)" python "$@"
