#!/usr/bin/env python3
"""
Launch Isaac Sim, load the XLERobot USD, add a ground plane, and drive the base with the keyboard.

Controls:
  W / S - forward / reverse
  A / D - rotate left / right
  Q / E - strafe left / right (simple omni-wheel approximation)
  Space  - brake (zero wheel velocities)
  Esc    - exit the simulator
"""

from __future__ import annotations

import os
from typing import Dict, Set

import numpy as np

from omni.isaac.kit import SimulationApp

# SimulationApp must be created before importing most Isaac Sim modules.
simulation_app = SimulationApp({"headless": False})

import carb
import omni.appwindow  # noqa: E402
import omni.kit.app  # noqa: E402
from carb.input import KeyboardEventType, KeyboardInput  # noqa: E402
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
from isaacsim.core.api.robots import Robot  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402


def _resolve_usd_path() -> str:
    """Return an absolute path to the robot USD that ships with this project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(current_dir, "xlerobot_wheel_v6.usd")
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"Could not find robot USD at {usd_path}")
    return usd_path


def _build_world(robot_usd: str) -> tuple[World, Robot]:
    """Create the Isaac Sim world with a ground plane and the XLERobot USD prim."""
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    world.scene.add(GroundPlane(prim_path="/World/GroundPlane", name="ground", size=50.0))
    add_reference_to_stage(robot_usd, "/World/XLERobot")
    robot = world.scene.add(
        Robot(
            prim_path="/World/XLERobot",
            name="xlerobot",
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # quaternion (w, x, y, z)
        )
    )
    world.reset()
    return world, robot


def _setup_keyboard_listener(active_keys: Set[str]):
    """Subscribe to the global keyboard event stream and update the active key cache."""
    input_interface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard_device = app_window.get_keyboard()

    def _on_key_event(event, *_):
        key_input = event.input
        if isinstance(key_input, KeyboardInput):
            key_name = key_input.name
        elif hasattr(key_input, "name"):
            key_name = key_input.name
        else:
            key_name = str(key_input).upper()
        if event.type == KeyboardEventType.KEY_PRESS or event.type == KeyboardEventType.KEY_REPEAT:
            active_keys.add(key_name)
        elif event.type == KeyboardEventType.KEY_RELEASE:
            active_keys.discard(key_name)
        if event.type == KeyboardEventType.KEY_PRESS and key_input == KeyboardInput.ESCAPE:
            simulation_app.close()
        return True

    subscription = input_interface.subscribe_to_keyboard_events(keyboard_device, _on_key_event)
    return subscription, input_interface


def _compute_wheel_velocities(active_keys: Set[str]) -> np.ndarray:
    """Translate the current keyboard state to wheel velocity targets."""
    base_speed = 15.0
    rotate_speed = 10.0
    strafe_speed = 12.0
    velocities = np.zeros(3, dtype=float)

    if "SPACE" in active_keys:
        return velocities

    if "W" in active_keys:
        velocities += base_speed
    if "S" in active_keys:
        velocities -= base_speed
    if "A" in active_keys:
        velocities += np.array([-rotate_speed, rotate_speed, -rotate_speed])
    if "D" in active_keys:
        velocities += np.array([rotate_speed, -rotate_speed, rotate_speed])
    if "Q" in active_keys:
        velocities += np.array([-strafe_speed, -strafe_speed, strafe_speed])
    if "E" in active_keys:
        velocities += np.array([strafe_speed, strafe_speed, -strafe_speed])

    return velocities


def main() -> None:
    robot_usd = _resolve_usd_path()
    world, robot = _build_world(robot_usd)

    joint_names = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
    joint_indices: Dict[str, int] = {name: robot.get_dof_index(name) for name in joint_names}
    articulation_view = robot._articulation_view  # noqa: SLF001 - direct access for control configuration
    for index in joint_indices.values():
        articulation_view.switch_dof_control_mode("velocity", dof_index=index)

    active_keys: Set[str] = set()
    keyboard_subscription, input_interface = _setup_keyboard_listener(active_keys)

    def _on_physics_step(step_size: float):
        velocities = _compute_wheel_velocities(active_keys)
        articulation_view.set_joint_velocity_targets(
            velocities[np.newaxis, :],
            joint_indices=np.array([joint_indices[name] for name in joint_names], dtype=np.int64),
        )

    world.add_physics_callback("xlerobot_keyboard_drive", _on_physics_step)

    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    finally:
        # Keep the subscription alive until shutdown, then release it explicitly.
        keyboard_subscription = None  # noqa: F841
        input_interface = None  # noqa: F841
        simulation_app.close()


if __name__ == "__main__":
    main()
