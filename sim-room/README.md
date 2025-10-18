# XLE Robot Room Simulation

This directory contains scripts and assets for loading a 3D room scene with the XLE robot in Isaac Sim.

## 📁 Files Overview

### 3D Scene Assets
- **`23Edit.usdz`** - Raw 3D scene exported from Gaussian Splatting reconstruction
- **`room_23.usd`** - Production-ready room scene with added physics (ground plane, table, mugs, colliders, lighting)
- **`xlerobot_wheel_v6.usd`** - XLE wheeled robot model (3-wheel omniwheel base with sensors)

### Shared Modules
- **`src/base_controller.py`** - Keyboard controller for 6-DOF omniwheel base movement (shared by both entry points)
- **`src/__init__.py`** - Package initialization

### Python Entry Points

#### `play_inRoom_sim.py` ⭐ **RECOMMENDED**
**Direct Isaac Sim approach** - Simple, high-quality rendering

- **Framework**: Isaac Sim only (no Isaac Lab)
- **Rendering Quality**: ✅ High (default RTX settings)
- **Stage Hierarchy**: Clean `/World/XLERobot` (no environment namespaces)
- **Code Complexity**: Low (~210 lines)
- **Keyboard Control**: ✅ Egocentric 6-DOF movement (W/A/S/D/Q/E)
- **Best For**: Visualization, prototyping, debugging
- **Usage**: `python sim-room/play_inRoom_sim.py`

**How it works:**
```python
# 1. Load room as base stage
omni.usd.get_context().open_stage("room_23.usd")

# 2. Add robot as reference
add_reference_to_stage(usd_path="robot.usd", prim_path="/World/XLERobot")

# 3. Direct USD API control
```

#### `play_inRoom_lab.py`
**Isaac Lab framework approach** - Optimized for RL training

- **Framework**: Isaac Lab (higher-level abstraction)
- **Rendering Quality**: ⚠️ Lower (optimized for RL performance)
- **Stage Hierarchy**: Nested `/World/envs/env_0/Room` and `/World/envs/env_0/XLERobot`
- **Code Complexity**: Medium-high (~220 lines)
- **Keyboard Control**: ✅ Egocentric 6-DOF movement (W/A/S/D/Q/E)
- **Best For**: Multi-environment RL training, complex asset management
- **Usage**: `python sim-room/play_inRoom_lab.py`

**How it works:**
```python
# 1. Define assets declaratively
scene_cfg = XleRoomSceneCfg(num_envs=1)
scene_cfg.room = AssetBaseCfg(...)
scene_cfg.robot = ArticulationCfg(...)

# 2. Use Isaac Lab scene management
scene = InteractiveScene(scene_cfg)
```

---

## 🎮 Keyboard Control

Both entry points support **egocentric 6-DOF omniwheel base control**:

### Movement Controls
- **W / ↑ / Numpad8**: Move Forward
- **S / ↓ / Numpad2**: Move Backward
- **A / ← / Numpad4**: Strafe Left
- **D / → / Numpad6**: Strafe Right

### Rotation Controls
- **Q / Numpad7**: Rotate Counter-Clockwise
- **E / Numpad9**: Rotate Clockwise

### Other
- **Space / Numpad5**: Stop All Movement

### Technical Implementation
- **Shared Controller**: `src/base_controller.py` contains `OmniBaseController` class used by both entry points
- **Omniwheel Kinematics**: Converts 6-DOF planar commands (forward/back, strafe left/right, rotate) to 3-wheel velocities
- **Isaac Sim Version**: Uses PhysX DriveAPI to set joint velocity targets
- **Isaac Lab Version**: Uses articulation API `set_joint_velocity_target()`

---

## 🎨 Rendering Quality Difference

### Why does rendering quality differ?

**Isaac Lab (`play_inRoom_lab.py`):**
- Programmatically **reduces rendering quality** for RL training performance
- Modifies RTX settings: Lower samples-per-pixel (SPP ~4-16), fewer light bounces (1-2), disabled accumulation
- **Optimized for**: Deterministic observations, fast physics simulation (60+ Hz)
- **Result**: Darker, lower quality, but consistent frame times

**Isaac Sim Direct (`play_inRoom_sim.py`):**
- Uses **default high-quality RTX rendering**
- Higher SPP (64-256), more light bounces (4-8), accumulation enabled
- **Optimized for**: Visual fidelity and human inspection
- **Result**: Better lighting, cleaner image quality

**Technical Root Cause:**
```python
# Isaac Lab's SimulationContext internally does:
carb.settings.set_int("/rtx/pathtracing/spp", 4)  # Reduce samples
carb.settings.set_bool("/rtx/pathtracing/accumulation/enabled", False)
carb.settings.set_int("/rtx/pathtracing/maxBounces", 1)  # Fewer bounces
```

---

## 🏗️ 3D Scene Generation Pipeline

### How to Create Real-World 3D Scenes

**Reference**: [NVIDIA Blog - Render Real-World Scenes](https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/)

**Tools Used:**
1. **COLMAP** - Structure from Motion (SfM) to get camera poses
2. **3D Gaussian Splatting** - Novel view synthesis and 3D reconstruction
3. **SuperSplat Editor** (https://superspl.at/editor/) - Clean up and export scene

**Workflow:**
```
📸 Photos/Video → COLMAP (SfM) → 3D Gaussian Splatting → SuperSplat → .usdz export
                                                                          ↓
                                                          23Edit.usdz (raw scene)
                                                                          ↓
                                        Add physics/colliders in Isaac Sim
                                                                          ↓
                                                          room_23.usd (production)
```

### Scene Preparation Steps:
1. Capture real-world scene with photos/video
2. Run COLMAP to get camera poses
3. Train 3D Gaussian Splatting model
4. Import to SuperSplat for cleanup
5. Export as USDZ
6. Import to Isaac Sim and add:
   - Ground plane collision
   - Table/furniture colliders
   - Physics materials
   - Lighting setup
7. Save as production USD (`room_23.usd`)

---

## 🚀 Quick Start

### Option 1: Simple Visualization (Recommended)
```bash
python sim-room/play_inRoom_sim.py
```
**Best for:** Quick testing, high-quality visualization, debugging robot in scene

### Option 2: Isaac Lab Framework
```bash
python sim-room/play_inRoom_lab.py
```
**Best for:** Setting up RL training pipeline, multi-environment support

### Command-line Options
```bash
# Simple version (no CLI args currently)
python sim-room/play_inRoom_sim.py

# Isaac Lab version supports:
python sim-room/play_inRoom_lab.py --robot-yaw 45.0  # Initial rotation in degrees
python sim-room/play_inRoom_lab.py --num_envs 1     # Number of environments
```

### Keyboard Controls
Once running, use **W/A/S/D** for movement, **Q/E** for rotation, and **Space** to stop. See the Keyboard Control section above for all available keys.

---

## 📊 Comparison Table

| Feature | `play_inRoom_sim.py` | `play_inRoom_lab.py` |
|---------|---------------------|-------------------|
| **Framework** | Isaac Sim only | Isaac Lab |
| **Rendering Quality** | ✅ High | ⚠️ Lower (RL-optimized) |
| **Stage Hierarchy** | Clean, flat | Nested (`/envs/env_0/`) |
| **Code Complexity** | Simple | Medium-High |
| **Keyboard Control** | ✅ Yes (PhysX DriveAPI) | ✅ Yes (Articulation API) |
| **Multi-Environment** | ❌ No | ✅ Yes (for RL) |
| **Initialization Speed** | Fast | Slower (framework overhead) |
| **Dependencies** | Minimal | Isaac Lab required |
| **Best Use Case** | **Visualization** | RL Training |

---

## 🎯 Which One to Use?

**Use `play_inRoom_sim.py` if:**
- You want high-quality rendering
- You're doing visualization/prototyping
- You have a single robot + single scene
- You want simple, readable code
- You need direct control with keyboard

**Use `play_inRoom_lab.py` if:**
- You're training RL agents
- You need multiple parallel environments
- You want Isaac Lab ecosystem integration
- You need advanced scene management features
- You're developing RL-based control policies

---

## 📝 Notes

- Both scripts load the same assets (`room_23.usd` + `xlerobot_wheel_v6.usd`)
- Both scripts use the same keyboard controller from `src/base_controller.py`
- The room already contains ground plane and lighting - no need for duplicates
- Robot is spawned at `(0, 0, 0)` by default
- Camera defaults to `eye=(3, 2, 2)`, `target=(0, 0, 1)`
- Use keyboard (W/A/S/D/Q/E) to control the robot base in 6 directions
- Press `Ctrl+C` or close window to exit

---

## 🔧 Troubleshooting

**"Module 'isaacsim' not found"**
- Make sure you're running from the Isaac Sim conda environment
- Check that Isaac Sim is properly installed

**"Robot appears dark when simulation is playing"**
- This is normal for the Isaac Lab version (RL optimization)
- Use `play_inRoom_sim.py` for better visual quality

**"Keyboard controls not working"**
- Make sure you're not in headless mode (check console output)
- Click on the Isaac Sim viewport to ensure it has focus
- Try pressing Space to reset the controller state

**"Robot not moving when keys are pressed"**
- Check console output for debug messages (printed every 30 frames)
- Verify the joint names match your robot USD (`axle_0_joint`, `axle_1_joint`, `axle_2_joint`)
- For Isaac Lab version, ensure actuators are properly configured

**"Camera view doesn't match expected position"**
- Camera API may vary between Isaac Sim versions
- Manually adjust viewport camera if automatic positioning fails
