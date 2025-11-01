#!/bin/bash

# Script to run Factory Environment Visualization Test
# This script tests the factory_version1 environment configuration

echo "=========================================="
echo "Factory Environment Visualization Test"
echo "=========================================="
echo ""

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$PWD/source/tacex_tasks:$PWD/source/tacex:$PWD/source/tacex_assets

# Default parameters
TASK="${1:-peg_insert}"
NUM_ENVS="${2:-1}"
PYTHON_BIN="/home/pi-zero/anaconda3/envs/env45_isaacsim/bin/python"

echo "Task: $TASK"
echo "Number of environments: $NUM_ENVS"
echo ""
echo "Available tasks: peg_insert, gear_mesh, nut_thread"
echo ""

# Run the visualization script
$PYTHON_BIN scripts/test_factory_visualization.py \
    --num_envs $NUM_ENVS \
    --task $TASK \
    --enable_cameras

echo ""
echo "Visualization test complete!"
echo ""
echo "To run with different tasks:"
echo "  ./run_factory_viz_test.sh peg_insert 1"
echo "  ./run_factory_viz_test.sh gear_mesh 1"
echo "  ./run_factory_viz_test.sh nut_thread 1"
echo ""
echo "To save tactile images, add --save_images flag manually"
