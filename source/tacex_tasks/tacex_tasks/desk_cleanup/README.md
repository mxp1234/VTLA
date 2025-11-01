# Desk Cleanup Task

This module implements a desk cleanup task where a Franka robot with GelSight tactile sensors must pick up various desktop objects and organize them.

## Overview

The desk cleanup environment simulates a realistic desk scenario with common desktop items that need to be organized. The robot must:
1. Identify objects on the desk
2. Grasp individual objects using tactile feedback
3. Move objects to designated placement locations
4. Release objects and return to ready position

## Features

- **Multiple Desktop Objects**: 6 different colored blocks (red, blue, green, yellow) and DexCube
- **No Bin**: Objects are placed directly on the desk surface
- **Tactile Feedback**: Real-time GelSight sensor visualization during grasping
- **Sequential Cleanup**: Grasp, lift, move, and place workflow
- **Configurable**: Adjustable number of objects and target object selection

## Files

```
source/tacex_tasks/tacex_tasks/desk_cleanup/
├── __init__.py                  # Module initialization
└── desk_cleanup_env_cfg.py      # Environment configuration

scripts/
└── desk_cleanup_demo.py         # Main demonstration script
```

## Configuration

### Scene Components

1. **Table**: Seattle Lab table (from ISAAC_NUCLEUS_DIR)
2. **Robot**: Franka Panda with GelSight Mini sensors on gripper
3. **Objects**: Colored blocks from Isaac Sim Props library
   - Red block
   - Blue block
   - Green block
   - Yellow block
   - DexCube (for dexterous manipulation)
   - Red block (duplicate)

### Object Layout

Objects are arranged on the desk in a compact pattern:
- Center area: Primary objects
- Spread pattern: Objects distributed for easy access
- All objects start at desk height (~5cm above table)

## Usage

### Basic Usage

Run with default settings (6 objects, target object 0):
```bash
python scripts/desk_cleanup_demo.py
```

### With GUI and Tactile Visualization

```bash
python scripts/desk_cleanup_demo.py --num_objects 6 --target_object 0
```

### Headless Mode

```bash
python scripts/desk_cleanup_demo.py --headless --num_objects 4 --target_object 1
```

### Parameters

- `--num_envs`: Number of parallel environments (default: 1)
- `--num_objects`: Number of objects to spawn, max 6 (default: 6)
- `--target_object`: Index of object to grasp first (default: 0)
- `--headless`: Run without GUI
- `--enable_cameras`: Enable camera sensors (automatically enabled)

## Cleanup Sequence

The robot follows this sequence for each object:

1. **IDLE**: Wait for scene initialization
2. **MOVE_TO_PREGRASP**: Move above the target object
3. **MOVE_TO_GRASP**: Lower to grasping height
4. **CLOSE_GRIPPER**: Close gripper to grasp object
5. **LIFT**: Lift object off the desk
6. **MOVE_TO_PLACE**: Move to placement location (left side of desk)
7. **RELEASE**: Open gripper to release object
8. **RETREAT**: Return to home position
9. **DONE**: Task complete

## Controls

When running with GUI:
- **'q'**: Quit simulation
- **'r'**: Reset scene and restart
- **Ctrl+C**: Force exit

## Tactile Visualization

Two OpenCV windows show real-time tactile sensor data:
- **GelSight Left**: Left gripper finger sensor
- **GelSight Right**: Right gripper finger sensor

The visualization displays:
- RGB tactile images (64x64 resolution)
- Contact deformation patterns
- Real-time updates during manipulation

## Customization

### Adding More Objects

To add custom objects, edit `desk_cleanup_env_cfg.py`:

```python
object_defs = [
    ("custom_object", f"{ISAAC_NUCLEUS_DIR}/Props/YourObject/object.usd",
     (scale_x, scale_y, scale_z), None, (offset_x, offset_y, offset_z)),
    # ... more objects
]
```

### Changing Object Positions

Modify the `desk_center` and offset values in `create_desk_objects()`:

```python
desk_center = np.array([0.5, 0.0, 0.05])  # [x, y, z]
offset = (x_offset, y_offset, z_offset)
```

### Adjusting Placement Location

Change the `placement_location` in `run_cleanup_demo()`:

```python
placement_location = torch.tensor([x, y, z], device=robot.device)
```

## Differences from Grasping Bin Environment

| Feature | Grasping Bin | Desk Cleanup |
|---------|--------------|--------------|
| Container | Bin present | No bin |
| Objects | Bolts and nuts | Colored blocks |
| Layout | Circular in bin | Scattered on desk |
| Task | Grasp and lift | Grasp, move, and place |
| Object Source | Factory assets | Isaac Sim Props |
| Placement | Lift only | Full pick-and-place |

## Troubleshooting

### Objects Fall Through Table
- Check that table and objects are at correct heights
- Verify collision properties in config

### Gripper Can't Grasp
- Adjust gripper approach angle
- Check object sizes match gripper opening
- Verify tactile sensor placement

### Tactile Windows Not Showing
- Ensure `--enable_cameras` is set (auto-enabled)
- Check that headless mode is disabled
- Verify GelSight sensors are initialized

## Future Enhancements

Potential improvements:
- [ ] Add more diverse objects (cups, pens, notebooks)
- [ ] Implement object classification
- [ ] Add placement targets/zones
- [ ] Multi-object sequential cleanup
- [ ] Obstacle avoidance during motion
- [ ] Dynamic object spawning

## References

- Based on `grasping_bin_env_cfg.py`
- Uses Isaac Sim Props library for objects
- Franka robot with GelSight Mini sensors from TacEx assets
