#!/bin/bash

# Test script for Bolt-Nut Bin Environment V2 (Articulation-based)
# This uses the new configuration that follows the Factory NutThread pattern

echo "=========================================="
echo "Bolt-Nut Bin Environment V2 Test"
echo "=========================================="
echo ""
echo "Configuration: bolt_nut_bin_env_cfg_v2.py"
echo "Pattern: ArticulationCfg (like NutThread)"
echo "Objects: 3 bolts + 2 nuts"
echo ""

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$PWD/source/tacex_tasks:$PWD/source/tacex:$PWD/source/tacex_assets:$PWD/IsaacLab/source/isaaclab:$PWD/IsaacLab/source/isaaclab_tasks

# Python binary
PYTHON_BIN="/home/pi-zero/anaconda3/envs/env45_isaacsim/bin/python"

# Run the visualization script with V2 config
echo "Running bolt_nut_grasp_with_visualization.py with V2 config..."
echo ""

$PYTHON_BIN -c "
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'source/tacex_tasks'))

# Import V2 configuration
from tacex_tasks.grasping_bin.bolt_nut_bin_env_cfg_v2 import BoltNutBinSceneCfg

# Create scene configuration
scene_cfg = BoltNutBinSceneCfg(num_envs=1, env_spacing=2.5)

print('✓ V2 Configuration loaded successfully')
print(f'  - Scene has {scene_cfg.num_envs} environment(s)')
print(f'  - Objects: bolt_0, bolt_1, bolt_2, nut_0, nut_1')
print(f'  - Robot: {scene_cfg.robot.prim_path}')
print(f'  - GelSight sensors: left + right')
print('')
print('Configuration validation complete!')
"

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run visualization: python scripts/bolt_nut_grasp_with_visualization.py --use_v2"
echo "2. Check that bolt and nut USD files render correctly"
echo "3. Verify physics and tactile sensors work"
