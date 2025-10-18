#!/usr/bin/env python3
"""
Load the indoor room scene, spawn the XLERobot, and drive the base with keyboard input.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict

import numpy as np

from omni.isaac.kit import SimulationApp

# SimulationApp must be created before importing most Isaac Sim modules.
simulation_app = SimulationApp({"headless": False})

import carb
import omni.appwindow  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api.robots import Robot  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

# Make sure shared controller module is importable.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "src"))
from base_controller import OmniBaseController  # noqa: E402


ROOM_USD_PATH = os.path.join(CURRENT_DIR, "room_23.usd")
ROBOT_USD_PATH = os.path.join(CURRENT_DIR, "xlerobot_wheel_v6.usd")
ROBOT_PRIM_PATH = "/World/XLERobot"
ROBOT_POSITION = np.array([0.0, 0.0, 0.0], dtype=float)
ROBOT_YAW_DEGREES = 0.0
CAMERA_EYE = (3.0, 2.0, 2.0)
CAMERA_TARGET = (0.0, 0.0, 1.0)


def _yaw_to_quaternion(yaw_degrees: float) -> np.ndarray:
    """Return a scalar-first quaternion (w, x, y, z) from a yaw angle in degrees."""
    half_angle = math.radians(yaw_degrees) * 0.5
    return np.array([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)], dtype=float)


def _set_camera_view(eye, target) -> None:
    """Position the perspective camera for a quick overview of the scene."""
    try:
        import omni.kit.viewport.utility as vp_util

        viewport = vp_util.get_active_viewport()
        if viewport:
            viewport.set_camera_position("/OmniverseKit_Persp", eye[0], eye[1], eye[2], True)
            viewport.set_camera_target("/OmniverseKit_Persp", target[0], target[1], target[2], True)
    except Exception as exc:  # pragma: no cover - best effort
        carb.log_warn(f"Unable to set camera view automatically: {exc}")


def main() -> None:
    if not os.path.exists(ROOM_USD_PATH):
        raise FileNotFoundError(f"Room USD not found: {ROOM_USD_PATH}")
    if not os.path.exists(ROBOT_USD_PATH):
        raise FileNotFoundError(f"Robot USD not found: {ROBOT_USD_PATH}")

    stage_ctx = omni.usd.get_context()
    stage_ctx.open_stage(ROOM_USD_PATH)

    # Build the Isaac Sim world on top of the loaded stage.
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)

    # Reference the robot USD into the stage and wrap it as a Robot.
    add_reference_to_stage(ROBOT_USD_PATH, ROBOT_PRIM_PATH)
    robot_orientation = _yaw_to_quaternion(ROBOT_YAW_DEGREES)
    robot = world.scene.add(
        Robot(
            prim_path=ROBOT_PRIM_PATH,
            name="xlerobot",
            position=ROBOT_POSITION,
            orientation=robot_orientation,
        )
    )

    # Reset to initialise physics handles for the robot and existing scene content.
    world.reset()

    # Configure wheel joints for velocity control.
    joint_names = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
    joint_indices: Dict[str, int] = {name: robot.get_dof_index(name) for name in joint_names}
    articulation_view = robot._articulation_view  # noqa: SLF001 - internal access to configure control
    for joint_index in joint_indices.values():
        articulation_view.switch_dof_control_mode("velocity", dof_index=joint_index)

    # Install the omni base keyboard controller.
    base_controller = OmniBaseController(base_speed=10.0, verbose=True)
    app_window = omni.appwindow.get_default_app_window()
    if app_window:
        base_controller.setup_keyboard(app_window)

    _set_camera_view(CAMERA_EYE, CAMERA_TARGET)

    def _on_physics_step(step_size: float) -> None:
        wheel_velocities = base_controller.get_wheel_velocities()
        articulation_view.set_joint_velocity_targets(
            wheel_velocities[np.newaxis, :],
            joint_indices=np.array([joint_indices[name] for name in joint_names], dtype=np.int64),
        )

    world.add_physics_callback("xlerobot_keyboard_drive", _on_physics_step)

    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    finally:
        base_controller.cleanup()
        simulation_app.close()


if __name__ == "__main__":
    main()

