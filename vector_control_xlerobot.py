# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Egocentric keyboard control for 3-wheel omniwheel base robot using kinematic equations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import carb
import omni.appwindow

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Get the current directory where the script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# ROBOT PHYSICAL PARAMETERS - CALIBRATE THESE FOR YOUR ROBOT
# ============================================================================
# Run measure_robot_dimensions.py to help determine these values
WHEEL_RADIUS = 0.05  # meters - radius of each omniwheel
BASE_RADIUS = 0.2    # meters - distance from robot center to wheel center

# Wheel angles (in degrees) - defines wheel placement
WHEEL_0_ANGLE = -90  # Back wheel
WHEEL_1_ANGLE = 150  # Front-left wheel
WHEEL_2_ANGLE = 30   # Front-right wheel

# Control parameters
BASE_SPEED = 0.5        # m/s - linear velocity multiplier
ROTATION_SPEED = 2.0    # rad/s - angular velocity multiplier
# ============================================================================

# Configuration for the omniwheel base
OMNI_BASE_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(CURRENT_DIR, "./robot_usd/xlerobot_wheel_cam.usd")),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            damping=None,  
            stiffness=None,
        ),
        # Add arm joints with high damping to keep them fixed
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw", 
                              "Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2",
                              "head_pan_joint", "head_tilt_joint"],
            damping=None,
            stiffness=None
        )
    },
)

class OmniBaseSceneCfg(InteractiveSceneCfg):
    """Designs the scene with omniwheel base."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    omni_base = OMNI_BASE_CONFIG.replace(prim_path="{ENV_REGEX_NS}/OmniBase")


def compute_omniwheel_velocities(velocity_cmd: torch.Tensor, 
                                 wheel_radius: float = WHEEL_RADIUS, 
                                 base_radius: float = BASE_RADIUS) -> torch.Tensor:
    """
    Compute wheel velocities for a 3-wheel omniwheel robot using kinematic equations.
    
    Assumes wheels are positioned at 120° intervals:
    - Wheel 0: -90° (back wheel)
    - Wheel 1: 150° (front-left)
    - Wheel 2:  30° (front-right)

    Args:
        velocity_cmd: Velocity command tensor [vx, vy, omega] where:
                      vx: Desired forward velocity (m/s)
                      vy: Desired lateral (strafe) velocity (m/s)
                      omega: Desired angular velocity (rad/s)
        wheel_radius: Radius of each wheel (m)
        base_radius: Distance from robot center to wheel center (m)
    
    Returns:
        torch.Tensor: Wheel velocities [w0, w1, w2] in rad/s
    """
    # Wheel angles (in radians) from robot's forward direction
    theta0 = np.deg2rad(WHEEL_0_ANGLE)
    theta1 = np.deg2rad(WHEEL_1_ANGLE)
    theta2 = np.deg2rad(WHEEL_2_ANGLE)
    
    # Inverse kinematics matrix for 3-wheel omniwheel
    # Each row represents one wheel's contribution
    # Matrix form: wheel_velocities = (1/r) * J * [vx, vy, omega]^T
    # where J is the Jacobian matrix
    
    # Build the Jacobian matrix (3x3)
    # Each row: [-cos(theta_i), sin(theta_i), -base_radius]
    J = torch.tensor([
        [-np.cos(theta0), np.sin(theta0), -base_radius],
        [-np.cos(theta1), np.sin(theta1), -base_radius],
        [-np.cos(theta2), np.sin(theta2), -base_radius]
    ], dtype=torch.float32)
    
    # Compute wheel velocities: w = (1/r) * J @ v
    wheel_velocities = (J @ velocity_cmd) / wheel_radius
    
    return wheel_velocities


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulation with egocentric keyboard control."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Base command tracking (like Spot demo)
    base_command = np.zeros(3)  # [forward/back, left/right, rotate]
    
    # Setup keyboard input
    appwindow = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    
    # Keyboard command mappings (like Spot demo)
    input_keyboard_mapping = {
        "W": [1.0, 0.0, 0.0], "UP": [1.0, 0.0, 0.0], "NUMPAD_8": [1.0, 0.0, 0.0],
        "S": [-1.0, 0.0, 0.0], "DOWN": [-1.0, 0.0, 0.0], "NUMPAD_2": [-1.0, 0.0, 0.0],
        "A": [0.0, 1.0, 0.0], "LEFT": [0.0, 1.0, 0.0], "NUMPAD_4": [0.0, 1.0, 0.0],
        "D": [0.0, -1.0, 0.0], "RIGHT": [0.0, -1.0, 0.0], "NUMPAD_6": [0.0, -1.0, 0.0],
        "Q": [0.0, 0.0, 1.0], "NUMPAD_7": [0.0, 0.0, 1.0],
        "E": [0.0, 0.0, -1.0], "NUMPAD_9": [0.0, 0.0, -1.0],
        "SPACE": 'stop', "NUMPAD_5": 'stop'
    }

    # Track currently pressed keys to avoid auto-repeat accumulating command multiple times
    pressed_keys = set()
    
    def keyboard_event_handler(event, *args, **kwargs):
        nonlocal base_command, pressed_keys
        # Some rare events may deliver a raw string instead of object
        key_name = event.input.name if hasattr(event.input, "name") else event.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in input_keyboard_mapping:
                # Ignore auto-repeat if key already held
                if key_name in pressed_keys:
                    return True
                pressed_keys.add(key_name)
                command = input_keyboard_mapping[key_name]
                if command == 'stop':
                    base_command = np.zeros(3)
                else:
                    base_command += np.array(command)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in input_keyboard_mapping:
                if key_name in pressed_keys:
                    pressed_keys.remove(key_name)
                command = input_keyboard_mapping[key_name]
                if command != 'stop':
                    base_command -= np.array(command)
        return True
    
    # Subscribe to keyboard events
    input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
    
    # Print controls
    print("\n" + "="*60)
    print("EGOCENTRIC KEYBOARD CONTROL - KINEMATIC MODEL")
    print("="*60)
    print("Movement Controls:")
    print("  W/↑/Numpad8     : Move Forward")
    print("  S/↓/Numpad2     : Move Backward")
    print("  A/←/Numpad4     : Strafe Left")
    print("  D/→/Numpad6     : Strafe Right")
    print("Rotation Controls:")
    print("  Q/Numpad7       : Rotate Counter-Clockwise")
    print("  E/Numpad9       : Rotate Clockwise")
    print("Other:")
    print("  Space/Numpad5   : Stop")
    print("="*60 + "\n")

    while simulation_app.is_running():
        # Reset environment periodically
        if count % 10000 == 0 and count > 0:
            count = 0
            root_state = scene["omni_base"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["omni_base"].write_root_pose_to_sim(root_state[:, :7])
            scene["omni_base"].write_root_velocity_to_sim(root_state[:, 7:])
            
            joint_pos, joint_vel = (
                scene["omni_base"].data.default_joint_pos.clone(),
                scene["omni_base"].data.default_joint_vel.clone(),
            )
            scene["omni_base"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Environment reset")
        
        # Calculate wheel velocities based on base command using kinematic model
        # Convert base command to velocity commands as a tensor
        velocity_cmd = torch.tensor([
            base_command[0] * BASE_SPEED,      # vx (m/s)
            base_command[1] * BASE_SPEED,      # vy (m/s)
            base_command[2] * ROTATION_SPEED   # omega (rad/s)
        ], dtype=torch.float32)
        
        # Compute wheel velocities using kinematic model
        wheel_velocities = compute_omniwheel_velocities(velocity_cmd)
        
        # Move to device
        wheel_velocities = wheel_velocities.to(scene["omni_base"].data.joint_pos.device)
        
        # Apply velocities to actuated joints
        actuator_joint_ids, _ = scene["omni_base"].find_joints(
            ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
        )

        # Convert to tensor if it's a list
        if isinstance(actuator_joint_ids, list):
            actuator_joint_ids = torch.tensor(actuator_joint_ids, device=scene["omni_base"].device, dtype=torch.long)

        # Debug output every 30 frames
        if count % 30 == 0:
            if np.any(base_command != 0):
                cmd_str = f"vx={velocity_cmd[0]:.3f}, vy={velocity_cmd[1]:.3f}, ω={velocity_cmd[2]:.3f}"
                vel_str = f"[{wheel_velocities[0]:.2f}, {wheel_velocities[1]:.2f}, {wheel_velocities[2]:.2f}]"
                print(f"Command: {cmd_str} | Wheel velocities: {vel_str} rad/s")
                # Also show which joints we're controlling
                if count % 300 == 0:  # Less frequent
                    print(f"  Controlling joint indices: {actuator_joint_ids.cpu().numpy()}")

        # Make sure we have valid joint IDs
        if len(actuator_joint_ids) == 3:
            scene["omni_base"].set_joint_velocity_target(
                wheel_velocities.unsqueeze(0),
                joint_ids=actuator_joint_ids
            )
        else:
            print(f"ERROR: Expected 3 joints but found {len(actuator_joint_ids)}")
        
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0/60.0)  # 60 Hz physics
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 3.5, 3.2], [0.0, 0.0, 0.5])
    
    # Design scene
    scene_cfg = OmniBaseSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
