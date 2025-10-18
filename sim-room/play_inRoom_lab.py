"""Launch Isaac Lab, load the XLERobot USD, and keep the sim running."""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from typing import Optional, Sequence, Union

from isaaclab.app import AppLauncher


# ---------------------------------------------------------------------------
# CLI / App launch (mirror official Isaac Lab examples to inherit defaults)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Load the custom XLERobot room scene inside Isaac Lab and play the simulation.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn (default: 1).",
)
parser.add_argument(
    "--robot-yaw",
    type=float,
    default=0.0,
    help="Initial yaw of the robot base in degrees (default: 0).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable viewport/window for visualization
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that require an active SimulationApp must happen after launch
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.assets.articulation import ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import UsdGeom  # noqa: E402

# Add src to path for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from base_controller import OmniBaseController  # noqa: E402
from xlerobot_dual_arm_ik import DualArmIKSolver  # noqa: E402

# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD_PATH = os.path.join(CURRENT_DIR, "xlerobot_wheel_v6.usd")
ROOM_USD_PATH = os.path.join(CURRENT_DIR, "room_23.usd")
CAMERA_EYE = (3.0, 2.0, 2.0)
CAMERA_TARGET = (0.0, 0.0, 1.0)
ROBOT_POS = (0.0, 0.0, 0.0)
ROOM_POS = (0.0, 0.0, 0.0)
TARGET_VISIBLE = False
TARGET_RADIUS = 0.05
TARGET_RIGHT_INIT = (0.45, 0.35, 1.1)
TARGET_LEFT_INIT = (0.45, -0.35, 1.1)

ROOM_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Room",  # Room inside env_0
    spawn=sim_utils.UsdFileCfg(usd_path=ROOM_USD_PATH),
    # Room uses its original position from the USD file
)

ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD_PATH),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            velocity_limit=100.0,
            effort_limit=10000.0,
            stiffness=None,  # Use None to let USD settings take precedence
            damping=None,    # Use None to let USD settings take precedence
        ),
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll",
                "Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2",
            ],
            damping=0.0,
            stiffness=1000.0,
        ),
    },
).replace(prim_path="{ENV_REGEX_NS}/XLERobot")  # Robot inside env_0, parallel to Room

TARGET_RIGHT_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/TargetRight",
    spawn=sim_utils.SphereCfg(
        radius=TARGET_RADIUS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.3)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=TARGET_RIGHT_INIT),
)

TARGET_LEFT_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/TargetLeft",
    spawn=sim_utils.SphereCfg(
        radius=TARGET_RADIUS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.6, 1.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=TARGET_LEFT_INIT),
)


def _yaw_to_quaternion(yaw_degrees: float) -> tuple[float, float, float, float]:
    """Convert yaw about +Z into a quaternion (w, x, y, z)."""
    half_rad = math.radians(yaw_degrees) * 0.5
    return (math.cos(half_rad), 0.0, 0.0, math.sin(half_rad))


class XleRoomSceneCfg(InteractiveSceneCfg):
    """Interactive scene configuration that spawns room and robot as parallel branches."""
    # Room USD already contains ground plane and lighting, so we don't duplicate them
    room = ROOM_CFG
    robot = ROBOT_CFG
    target_right = TARGET_RIGHT_CFG
    target_left = TARGET_LEFT_CFG


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _set_prim_visibility(target: Union[str, Sequence[str], object], visible: bool, scene: Optional[InteractiveScene] = None) -> None:
    """Toggle visibility for USD prims referenced by name, path, or asset."""

    def _iter_paths(item: Union[str, Sequence[str], object]):
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                yield from _iter_paths(sub_item)
        elif hasattr(item, "prim_paths"):
            yield from _iter_paths(getattr(item, "prim_paths"))
        elif isinstance(item, str):
            if item.startswith("/"):
                yield item
            elif scene is not None:
                try:
                    asset = scene[item]
                except KeyError:
                    print(f"[WARN] Prim name not found in scene: {item}")
                else:
                    yield from _iter_paths(asset.prim_paths)
            else:
                print(f"[WARN] Cannot resolve prim name without scene: {item}")
        else:
            print(f"[WARN] Unsupported target for visibility toggle: {item}")

    stage = omni.usd.get_context().get_stage()
    for prim_path in _iter_paths(target):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            print(f"[WARN] Prim not found: {prim_path}")
            continue
        img = UsdGeom.Imageable(prim)
        if visible:
            img.MakeVisible()
        else:
            img.MakeInvisible()


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.isfile(ROBOT_USD_PATH):
        raise FileNotFoundError(f"Robot USD not found: {ROBOT_USD_PATH}")
    if not os.path.isfile(ROOM_USD_PATH):
        raise FileNotFoundError(f"Room USD not found: {ROOM_USD_PATH}")

    print(f"[play_xle_room] Using simulation device: {args_cli.device}")
    print(f"[play_xle_room] Robot USD path: {ROBOT_USD_PATH}")
    print(f"[play_xle_room] Room USD path: {ROOM_USD_PATH}")
    print(f"[play_xle_room] Creating simulation context...")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    print(f"[play_xle_room] Setting camera view...")
    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

    print(f"[play_xle_room] Creating scene with room and robot...")
    try:
        # Create scene config with both room and robot
        scene_cfg = XleRoomSceneCfg(num_envs=1, env_spacing=0.0)
        scene_cfg.robot.init_state.pos = ROBOT_POS
        scene_cfg.robot.init_state.rot = _yaw_to_quaternion(args_cli.robot_yaw)

        print(f"[play_xle_room] Scene config created, instantiating InteractiveScene...")
        scene = InteractiveScene(scene_cfg)
        print(f"[play_xle_room] Scene created successfully.")
    except Exception as exc:
        import traceback as tb
        print(f"[play_xle_room] ERROR: Failed to create scene: {exc}")
        tb.print_exc()
        raise

    try:
        print(f"[play_xle_room] Resetting simulation...")
        sim.reset()
        print(f"[play_xle_room] Simulation reset complete.")
    except Exception as exc:
        import traceback
        print("[play_xle_room] sim.reset() failed:", exc)
        traceback.print_exc()
        raise

    print("[play_xle_room] Resetting scene to default state...")
    scene.reset()
    print("[play_xle_room] Scene reset complete.")

    print("[play_xle_room] Writing scene data to sim...")
    scene.write_data_to_sim()
    print("[play_xle_room] Scene data written.")

    print("[play_xle_room] Configuring IK target visibility...")
    _set_prim_visibility(scene["target_left"], TARGET_VISIBLE, scene)
    _set_prim_visibility(scene["target_right"], TARGET_VISIBLE, scene)

    print("\n[play_xle_room] Assets loaded. Isaac Lab simulation running.")
    print("  Room USD:  ", ROOM_USD_PATH)
    print("  Robot USD: ", ROBOT_USD_PATH)

    # Setup keyboard control
    print("[play_xle_room] Setting up keyboard control...")
    base_controller = OmniBaseController(base_speed=10.0, verbose=True)
    appwindow = omni.appwindow.get_default_app_window()
    if appwindow and base_controller.setup_keyboard(appwindow):
        print("[play_xle_room] ✓ Keyboard control enabled")
    else:
        print("[play_xle_room] ⚠ Keyboard control unavailable (headless mode)")

    print("Close the Isaac Lab window or press Ctrl+C to exit.\n")

    sim_dt = sim.get_physics_dt()
    frame = 0

    # Get device for tensor operations
    device = sim.device
    print("[play_xle_room] Initializing dual-arm IK solver...")
    ik_solver = DualArmIKSolver(
        robot=scene["robot"],
        left_target=scene["target_left"],
        right_target=scene["target_right"],
        device=device,
    )
    attach_cooldown = 0.0
    was_moving = False

    # Find the joint indices for the wheel joints
    robot_joint_names = scene["robot"].data.joint_names
    wheel_joint_names = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
    wheel_joint_indices = []
    for wheel_name in wheel_joint_names:
        try:
            idx = robot_joint_names.index(wheel_name)
            wheel_joint_indices.append(idx)
        except ValueError:
            print(f"[play_xle_room] Warning: Joint '{wheel_name}' not found in robot")
    wheel_joint_ids = None
    if len(wheel_joint_indices) == 3:
        wheel_joint_ids = torch.tensor(wheel_joint_indices, dtype=torch.long, device=device)

    print(f"[play_xle_room] Robot joint names: {robot_joint_names}")
    print(f"[play_xle_room] Wheel joint indices: {wheel_joint_indices}")
    print(f"[play_xle_room] Number of actuated joints: {scene['robot'].num_joints}")

    # Check if actuators are configured
    if hasattr(scene["robot"], 'actuators'):
        print(f"[play_xle_room] Robot actuators: {scene['robot'].actuators}")
        for name, actuator in scene["robot"].actuators.items():
            print(f"[play_xle_room]   - {name}: {actuator.joint_names}")
    else:
        print(f"[play_xle_room] WARNING: Robot has no actuators configured!")

    # Ensure initial update to show the window
    simulation_app.update()
    print("[play_xle_room] Window should now be visible. Starting simulation loop...")

    try:
        while simulation_app.is_running():
            base_moving = base_controller.is_moving()
            if base_moving:
                attach_cooldown = 0.5
                was_moving = True
            else:
                attach_cooldown = max(0.0, attach_cooldown - sim_dt)
                if was_moving and attach_cooldown == 0.0:
                    ik_solver.reset()
                    was_moving = False
                    print("[play_xle_room] Base stopped - IK solver reset")

            # Get wheel velocities from keyboard control
            wheel_velocities = base_controller.get_wheel_velocities(device=device)

            # Apply velocities to robot wheels using Isaac Lab articulation API
            if wheel_joint_ids is not None:
                joint_vel_target = wheel_velocities.unsqueeze(0)  # Shape: [1, 3] for batch dimension
                scene["robot"].set_joint_velocity_target(joint_vel_target, joint_ids=wheel_joint_ids)

                if frame % 60 == 0 and torch.any(wheel_velocities != 0):
                    print(f"[Debug Frame {frame}] Set velocity target: {joint_vel_target.cpu().numpy()}")
                    current_vel = scene["robot"].data.joint_vel[0, wheel_joint_indices].cpu().numpy()
                    print(f"[Debug Frame {frame}] Current joint velocities: {current_vel}")

            if base_moving or attach_cooldown > 0.0:
                ik_solver.sync_targets_to_current_pose()

            ik_solver.step()

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            simulation_app.update()

            frame += 1
            base_controller.print_debug_info(frame, wheel_velocities)

            if frame % 120 == 0:
                print(f"[play_xle_room] Running... ({frame} frames)")

    except KeyboardInterrupt:
        print("\n[play_xle_room] Interrupted by user")
    finally:
        base_controller.cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[play_xle_room] FATAL ERROR: {e}")
        traceback.print_exc()
        import sys
        sys.exit(1)
    finally:
        print("[play_xle_room] Closing simulation app...")
        simulation_app.close()
