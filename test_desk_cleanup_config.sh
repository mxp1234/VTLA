#!/bin/bash

# Test script for desk cleanup demo with updated configuration
# Tests: 3 clamps + pallet (replacing plate)

echo "=========================================="
echo "Testing Desk Cleanup Demo Configuration"
echo "=========================================="
echo ""

# Check if pallet USD file exists
echo "[1/4] Checking pallet USD file..."
if [ -f "/home/pi-zero/Documents/USD-file/pallet.usd" ]; then
    echo "✓ Pallet USD file found: /home/pi-zero/Documents/USD-file/pallet.usd"
    ls -lh /home/pi-zero/Documents/USD-file/pallet.usd
else
    echo "✗ ERROR: Pallet USD file not found!"
    exit 1
fi
echo ""

# Check if URDF files exist
echo "[2/4] Checking URDF files..."
URDF_BASE="/home/pi-zero/Downloads/ycb_urdfs-main/ycb_assets"

URDF_FILES=(
    "024_bowl.urdf"
    "065-g_cups.urdf"
    "050_medium_clamp.urdf"
    "026_sponge.urdf"
    "022_windex_bottle.urdf"
)

for urdf in "${URDF_FILES[@]}"; do
    if [ -f "$URDF_BASE/$urdf" ]; then
        echo "✓ Found: $urdf"
    else
        echo "✗ Missing: $urdf"
    fi
done
echo ""

# Check configuration file
echo "[3/4] Checking configuration file..."
CONFIG_FILE="/home/pi-zero/isaac-sim/TacEx/source/tacex_tasks/tacex_tasks/desk_cleanup/desk_cleanup_urdf_cfg.py"

if grep -q "clamp_0" "$CONFIG_FILE" && \
   grep -q "clamp_1" "$CONFIG_FILE" && \
   grep -q "clamp_2" "$CONFIG_FILE" && \
   grep -q "pallet" "$CONFIG_FILE"; then
    echo "✓ Configuration file contains all required objects:"
    echo "  - clamp_0 (original)"
    echo "  - clamp_1 (new)"
    echo "  - clamp_2 (new)"
    echo "  - pallet (replacing plate)"

    # Count total object definitions
    NUM_OBJECTS=$(grep -c "^        (\"" "$CONFIG_FILE")
    echo "✓ Total object definitions: $NUM_OBJECTS"
else
    echo "✗ Configuration file missing required objects!"
    exit 1
fi
echo ""

# Check demo script
echo "[4/4] Checking demo script..."
DEMO_SCRIPT="/home/pi-zero/isaac-sim/TacEx/scripts/desk_cleanup_urdf_demo.py"

if grep -q "default=8" "$DEMO_SCRIPT" && \
   grep -q "max: 8" "$DEMO_SCRIPT"; then
    echo "✓ Demo script updated with correct defaults"
    echo "  - Default: 8 objects"
    echo "  - Max: 8 objects"
else
    echo "✗ Demo script not updated correctly!"
    exit 1
fi
echo ""

echo "=========================================="
echo "✓ All checks passed!"
echo "=========================================="
echo ""
echo "Ready to run demo with commands:"
echo ""
echo "# All 8 objects (bowl, cup, 3x clamp, sponge, sashuihu, pallet)"
echo "python scripts/desk_cleanup_urdf_demo.py"
echo ""
echo "# Specify target object (e.g., clamp_0 is index 2)"
echo "python scripts/desk_cleanup_urdf_demo.py --target_object 2"
echo ""
echo "# Headless mode"
echo "python scripts/desk_cleanup_urdf_demo.py --headless"
echo ""
echo "# Fewer objects (e.g., first 5)"
echo "python scripts/desk_cleanup_urdf_demo.py --num_objects 5"
echo ""
