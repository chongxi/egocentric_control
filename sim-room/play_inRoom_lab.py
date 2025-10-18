"""Launch Isaac Lab, load the XLERobot USD, and keep the sim running."""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
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
import torch  # noqa: E402

# Add src to path for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from base_controller import OmniBaseController  # noqa: E402

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
    },
).replace(prim_path="{ENV_REGEX_NS}/XLERobot")  # Robot inside env_0, parallel to Room


def _yaw_to_quaternion(yaw_degrees: float) -> tuple[float, float, float, float]:
    """Convert yaw about +Z into a quaternion (w, x, y, z)."""
    half_rad = math.radians(yaw_degrees) * 0.5
    return (math.cos(half_rad), 0.0, 0.0, math.sin(half_rad))


class XleRoomSceneCfg(InteractiveSceneCfg):
    """Interactive scene configuration that spawns room and robot as parallel branches."""
    # Room USD already contains ground plane and lighting, so we don't duplicate them
    room = ROOM_CFG
    robot = ROBOT_CFG


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
            # Get wheel velocities from keyboard control
            wheel_velocities = base_controller.get_wheel_velocities(device=device)

            # Apply velocities to robot wheels using Isaac Lab articulation API
            # The robot has 3 wheel joints: axle_0_joint, axle_1_joint, axle_2_joint
            if len(wheel_joint_indices) == 3:
                # Set joint velocity targets for the 3 wheel joints
                # wheel_velocities is shape [3] for the 3 wheels
                joint_vel_target = wheel_velocities.unsqueeze(0)  # Shape: [1, 3] for batch dimension
                scene["robot"].set_joint_velocity_target(joint_vel_target, joint_ids=wheel_joint_indices)

                # Debug: Check if the target was set correctly
                if frame % 60 == 0 and torch.any(wheel_velocities != 0):
                    print(f"[Debug Frame {frame}] Set velocity target: {joint_vel_target.cpu().numpy()}")
                    print(f"[Debug Frame {frame}] Current joint velocities: {scene['robot'].data.joint_vel[0, wheel_joint_indices].cpu().numpy()}")

            # Write scene data (includes both room and robot)
            scene.write_data_to_sim()

            # Step physics
            sim.step()

            # Update scene (includes both room and robot)
            scene.update(sim_dt)

            # Update viewer
            simulation_app.update()

            frame += 1

            # Debug output
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
