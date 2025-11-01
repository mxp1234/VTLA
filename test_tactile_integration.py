#!/usr/bin/env python3
"""
Test script to verify tactile force field integration in factory environment.

Usage:
    cd /home/pi-zero/isaac-sim/TacEx
    PYTHONPATH=$PYTHONPATH:$PWD/source/tacex_tasks:$PWD/source/tacex:$PWD/source/tacex_assets \
    python test_tactile_integration.py
"""

import torch
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), "source/tacex_tasks"))
sys.path.insert(0, os.path.join(os.getcwd(), "source/tacex"))
sys.path.insert(0, os.path.join(os.getcwd(), "source/tacex_assets"))

from tacex_tasks.factory_version1.network.tactile_feature_extractor import create_tactile_encoder


def test_tactile_extractor():
    """Test the tactile force field extractor standalone."""
    print("=" * 80)
    print("Testing Tactile Force Field Extractor")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Checkpoint path
    checkpoint_path = os.path.join(
        os.getcwd(),
        "source/tacex_tasks/tacex_tasks/factory_version1/network/last.ckpt"
    )

    if not os.path.exists(checkpoint_path):
        print(f"\n❌ ERROR: Checkpoint not found at {checkpoint_path}")
        return False

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Checkpoint size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")

    # Create extractor
    print("\n1. Creating tactile extractor...")
    try:
        extractor = create_tactile_encoder(
            encoder_type='force_field',
            checkpoint_path=checkpoint_path,
            device=device,
            freeze_model=True
        )
        print("✓ Extractor created successfully")
    except Exception as e:
        print(f"❌ ERROR creating extractor: {e}")
        return False

    # Test inference
    print("\n2. Testing inference...")
    batch_size = 4
    height, width = 224, 224

    # Create dummy tactile data (normalized to [0, 1])
    tactile_left = torch.rand(batch_size, height, width, 3).to(device)
    tactile_right = torch.rand(batch_size, height, width, 3).to(device)

    print(f"   Input shape (left): {tactile_left.shape}")
    print(f"   Input shape (right): {tactile_right.shape}")
    print(f"   Input range: [{tactile_left.min():.3f}, {tactile_left.max():.3f}]")

    try:
        with torch.no_grad():
            features = extractor(tactile_left, tactile_right)

        print(f"\n   Output shape: {features.shape}")
        print(f"   Output dim: {extractor.get_output_dim()}")
        print(f"   Expected shape: ({batch_size}, 3)")

        if features.shape != (batch_size, 3):
            print(f"❌ ERROR: Unexpected output shape {features.shape}")
            return False

        print(f"\n   Sample features (first batch):")
        print(f"     normal_sum: {features[0, 0]:.4f}")
        print(f"     shear_x_sum: {features[0, 1]:.4f}")
        print(f"     shear_y_sum: {features[0, 2]:.4f}")

        print("\n✓ Inference test passed")

    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with uint8 data (simulating sensor output)
    print("\n3. Testing with uint8 sensor data...")
    tactile_left_uint8 = torch.randint(0, 256, (batch_size, height, width, 3), dtype=torch.uint8).to(device)
    tactile_right_uint8 = torch.randint(0, 256, (batch_size, height, width, 3), dtype=torch.uint8).to(device)

    # Normalize to [0, 1]
    tactile_left_norm = tactile_left_uint8.float() / 255.0
    tactile_right_norm = tactile_right_uint8.float() / 255.0

    print(f"   Input range (before norm): [{tactile_left_uint8.min()}, {tactile_left_uint8.max()}]")
    print(f"   Input range (after norm): [{tactile_left_norm.min():.3f}, {tactile_left_norm.max():.3f}]")

    try:
        with torch.no_grad():
            features = extractor(tactile_left_norm, tactile_right_norm)

        print(f"   Output shape: {features.shape}")
        print("✓ uint8 data test passed")

    except Exception as e:
        print(f"❌ ERROR with uint8 data: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    return True


def test_observation_dimensions():
    """Test that observation dimensions match configuration."""
    print("\n" + "=" * 80)
    print("Testing Observation Dimensions")
    print("=" * 80)

    from tacex_tasks.factory_version1.factory_env_cfg import OBS_DIM_CFG, FactoryTaskPegInsertCfg

    cfg = FactoryTaskPegInsertCfg()

    print("\nConfigured obs_order:")
    for i, obs_name in enumerate(cfg.obs_order, 1):
        dim = OBS_DIM_CFG[obs_name]
        print(f"  {i}. {obs_name}: {dim}")

    total_obs_dim = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
    print(f"\nTotal observation dimension (without prev_actions): {total_obs_dim}")
    print(f"Total observation dimension (with prev_actions): {total_obs_dim + cfg.action_space}")

    # Verify tactile_force_field is in obs_order
    if "tactile_force_field" in cfg.obs_order:
        print("\n✓ tactile_force_field found in obs_order")
        print(f"  Dimension: {OBS_DIM_CFG['tactile_force_field']}")
    else:
        print("\n❌ ERROR: tactile_force_field not in obs_order")
        return False

    # Verify raw tactile images are NOT in obs_order (only in state_order)
    if "tactile_left" not in cfg.obs_order and "tactile_right" not in cfg.obs_order:
        print("✓ Raw tactile images correctly removed from obs_order")
    else:
        print("❌ ERROR: Raw tactile images still in obs_order")
        return False

    print("\n" + "=" * 80)
    print("✓ Observation dimension test passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TACTILE FORCE FIELD INTEGRATION TEST")
    print("=" * 80)

    success = True

    # Test 1: Tactile extractor
    if not test_tactile_extractor():
        success = False

    # Test 2: Observation dimensions
    if not test_observation_dimensions():
        success = False

    # Summary
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Run the factory environment to verify integration")
        print("2. Start RL training with tactile force field features")
        print("3. Monitor training performance")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check the error messages above")
    print("=" * 80 + "\n")

    sys.exit(0 if success else 1)
