"""Launch Isaac Lab, load the XLERobot USD, and keep the sim running."""

from __future__ import annotations

import argparse
import math
import os
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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402

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
    actuators={},  # No actuators needed for passive/kinematic viewing
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
    print("Close the Isaac Lab window or press Ctrl+C to exit.\n")

    sim_dt = sim.get_physics_dt()
    frame = 0

    # Ensure initial update to show the window
    simulation_app.update()
    print("[play_xle_room] Window should now be visible. Starting simulation loop...")

    while simulation_app.is_running():
        # Write scene data (includes both room and robot)
        scene.write_data_to_sim()

        # Step physics
        sim.step()

        # Update scene (includes both room and robot)
        scene.update(sim_dt)

        # Update viewer
        simulation_app.update()

        frame += 1
        if frame % 120 == 0:
            print(f"[play_xle_room] Running... ({frame} frames)")


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
