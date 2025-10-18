# XLE Robot Room Simulation

This directory contains scripts and assets for loading a 3D room scene with the XLE robot in Isaac Sim.

## 📁 Files Overview

### 3D Scene Assets
- **`23Edit.usdz`** - Raw 3D scene exported from Gaussian Splatting reconstruction
- **`room_23.usd`** - Production-ready room scene with added physics (ground plane, table, mugs, colliders, lighting)
- **`xlerobot_wheel_v6.usd`** - XLE wheeled robot model (3-wheel omniverse base with sensors)

### Python Entry Points

#### `play_xle_simple.py` ⭐ **RECOMMENDED**
**Direct Isaac Sim approach** - Simple, high-quality rendering

- **Framework**: Isaac Sim only (no Isaac Lab)
- **Rendering Quality**: ✅ High (default RTX settings)
- **Stage Hierarchy**: Clean `/World/XLERobot` (no environment namespaces)
- **Code Complexity**: Low (~170 lines)
- **Best For**: Visualization, prototyping, debugging
- **Usage**: `python sim-room/play_xle_simple.py`

**How it works:**
```python
# 1. Load room as base stage
omni.usd.get_context().open_stage("room_23.usd")

# 2. Add robot as reference
add_reference_to_stage(usd_path="robot.usd", prim_path="/World/XLERobot")

# 3. Direct USD API control
```

#### `play_xle_room.py`
**Isaac Lab framework approach** - Optimized for RL training

- **Framework**: Isaac Lab (higher-level abstraction)
- **Rendering Quality**: ⚠️ Lower (optimized for RL performance)
- **Stage Hierarchy**: Nested `/World/envs/env_0/Room` and `/World/envs/env_0/XLERobot`
- **Code Complexity**: Medium-high (~177 lines)
- **Best For**: Multi-environment RL training, complex asset management
- **Usage**: `python sim-room/play_xle_room.py`

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

## 🎨 Rendering Quality Difference

### Why does rendering quality differ?

**Isaac Lab (`play_xle_room.py`):**
- Programmatically **reduces rendering quality** for RL training performance
- Modifies RTX settings: Lower samples-per-pixel (SPP ~4-16), fewer light bounces (1-2), disabled accumulation
- **Optimized for**: Deterministic observations, fast physics simulation (60+ Hz)
- **Result**: Darker, lower quality, but consistent frame times

**Isaac Sim Direct (`play_xle_simple.py`):**
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
python sim-room/play_xle_simple.py
```
**Best for:** Quick testing, high-quality visualization, debugging robot in scene

### Option 2: Isaac Lab Framework
```bash
python sim-room/play_xle_room.py
```
**Best for:** Setting up RL training pipeline, multi-environment support

### Command-line Options
```bash
# Set robot initial yaw
python sim-room/play_xle_simple.py  # (Simple version has no CLI args yet)

# Isaac Lab version supports:
python sim-room/play_xle_room.py --robot-yaw 45.0  # Initial rotation in degrees
python sim-room/play_xle_room.py --num_envs 1     # Number of environments
```

---

## 📊 Comparison Table

| Feature | `play_xle_simple.py` | `play_xle_room.py` |
|---------|---------------------|-------------------|
| **Framework** | Isaac Sim only | Isaac Lab |
| **Rendering Quality** | ✅ High | ⚠️ Lower (RL-optimized) |
| **Stage Hierarchy** | Clean, flat | Nested (`/envs/env_0/`) |
| **Code Complexity** | Simple | Medium-High |
| **Multi-Environment** | ❌ No | ✅ Yes (for RL) |
| **Initialization Speed** | Fast | Slower (framework overhead) |
| **Dependencies** | Minimal | Isaac Lab required |
| **Best Use Case** | **Visualization** | RL Training |

---

## 🎯 Which One to Use?

**Use `play_xle_simple.py` if:**
- You want high-quality rendering
- You're doing visualization/prototyping
- You have a single robot + single scene
- You want simple, readable code

**Use `play_xle_room.py` if:**
- You're training RL agents
- You need multiple parallel environments
- You want Isaac Lab ecosystem integration
- You need advanced scene management features

---

## 📝 Notes

- Both scripts load the same assets (`room_23.usd` + `xlerobot_wheel_v6.usd`)
- The room already contains ground plane and lighting - no need for duplicates
- Robot is spawned at `(0, 0, 0)` by default
- Camera defaults to `eye=(3, 2, 2)`, `target=(0, 0, 1)`
- Press `Ctrl+C` or close window to exit

---

## 🔧 Troubleshooting

**"Module 'isaacsim' not found"**
- Make sure you're running from the Isaac Sim conda environment
- Check that Isaac Sim is properly installed

**"Robot appears dark when simulation is playing"**
- This is normal for the Isaac Lab version (RL optimization)
- Use `play_xle_simple.py` for better visual quality

**"Camera view doesn't match expected position"**
- Camera API may vary between Isaac Sim versions
- Manually adjust viewport camera if automatic positioning fails
