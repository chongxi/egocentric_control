"""Simple approach: Load room_23.usd as base stage, then add robot as reference."""

import os
import sys
from isaacsim import SimulationApp

# Launch Isaac Sim with viewer
simulation_app = SimulationApp(launch_config={"headless": False})

import omni.kit.commands
import omni.usd
import omni.timeline
import omni.appwindow
import omni.physx
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics

# Try to import ArticulationView from omni.isaac.core
try:
    from omni.isaac.core.articulations import ArticulationView
    ARTICULATION_VIEW_AVAILABLE = True
except ImportError:
    ARTICULATION_VIEW_AVAILABLE = False
    ArticulationView = None

# Add src to path for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from base_controller import OmniBaseController

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOM_USD_PATH = os.path.join(CURRENT_DIR, "room_23.usd")
ROBOT_USD_PATH = os.path.join(CURRENT_DIR, "xlerobot_wheel_v6.usd")

# Robot spawn configuration
ROBOT_PRIM_PATH = "/World/XLERobot"
ROBOT_POSITION = (0.0, 0.0, 0.0)  # Position in the room
ROBOT_YAW_DEGREES = 0.0  # Initial yaw rotation

# Camera configuration
CAMERA_EYE = (3.0, 2.0, 2.0)
CAMERA_TARGET = (0.0, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def set_camera_view(eye, target):
    """Set the viewport camera position and target."""
    try:
        import omni.kit.viewport.utility as vp_util
        viewport_api = vp_util.get_active_viewport()
        if viewport_api:
            viewport_api.set_camera_position(
                "/OmniverseKit_Persp",
                eye[0], eye[1], eye[2],
                True
            )
            viewport_api.set_camera_target(
                "/OmniverseKit_Persp",
                target[0], target[1], target[2],
                True
            )
    except Exception as e:
        print(f"         Warning: Could not set camera view: {e}")
        print(f"         (Camera positioning may not work - you can manually adjust)")


def yaw_to_quaternion(yaw_degrees):
    """Convert yaw (in degrees) to quaternion (w, x, y, z)."""
    import math
    half_rad = math.radians(yaw_degrees) * 0.5
    return Gf.Quatd(math.cos(half_rad), 0.0, 0.0, math.sin(half_rad))  # Use Quatd (double precision)


# ---------------------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("XLE Robot Room Scene - Simple Loading Method")
    print("="*70 + "\n")

    # Validate files exist
    if not os.path.exists(ROOM_USD_PATH):
        print(f"ERROR: Room USD not found: {ROOM_USD_PATH}")
        return
    if not os.path.exists(ROBOT_USD_PATH):
        print(f"ERROR: Robot USD not found: {ROBOT_USD_PATH}")
        return

    # Step 1: Open the room as the base stage
    print(f"[Step 1] Opening room stage: {ROOM_USD_PATH}")
    omni.usd.get_context().open_stage(ROOM_USD_PATH)
    stage = omni.usd.get_context().get_stage()
    print(f"         ✓ Stage opened successfully")
    print(f"         Stage has {len(list(stage.Traverse()))} prims")

    # Step 2: Add robot as a reference to the stage
    print(f"\n[Step 2] Adding robot as reference: {ROBOT_USD_PATH}")
    print(f"         Spawning at path: {ROBOT_PRIM_PATH}")
    robot_prim = add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)

    if not robot_prim:
        print(f"ERROR: Failed to add robot reference")
        return
    print(f"         ✓ Robot reference added successfully")

    # Step 3: Set robot position
    print(f"\n[Step 3] Setting robot transform")
    print(f"         Position: {ROBOT_POSITION}")
    print(f"         Yaw: {ROBOT_YAW_DEGREES}°")

    # Add translate operation if it doesn't exist
    if not robot_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(robot_prim).AddTranslateOp()
    robot_prim.GetAttribute("xformOp:translate").Set(ROBOT_POSITION)

    # Add rotation operation if it doesn't exist
    if not robot_prim.GetAttribute("xformOp:orient"):
        UsdGeom.Xformable(robot_prim).AddOrientOp()
    robot_quat = yaw_to_quaternion(ROBOT_YAW_DEGREES)
    robot_prim.GetAttribute("xformOp:orient").Set(robot_quat)
    print(f"         ✓ Transform applied")

    # Step 4: Set camera view
    print(f"\n[Step 4] Setting camera view")
    print(f"         Eye: {CAMERA_EYE}")
    print(f"         Target: {CAMERA_TARGET}")
    set_camera_view(CAMERA_EYE, CAMERA_TARGET)
    print(f"         ✓ Camera positioned")

    # Step 5: Display stage structure
    print(f"\n[Step 5] Stage Structure:")
    print("         /World")
    for prim in stage.GetPrimAtPath("/World").GetChildren():
        prim_type = prim.GetTypeName()
        prim_name = prim.GetName()
        print(f"           ├── /{prim_name} ({prim_type})")
        if prim_name in ["Room", "XLERobot"]:
            for child in prim.GetChildren()[:3]:  # Show first 3 children
                child_type = child.GetTypeName()
                child_name = child.GetName()
                print(f"           │   ├── {child_name} ({child_type})")

    # Step 6: Setup keyboard control
    print(f"\n[Step 6] Setting up keyboard control")
    base_controller = OmniBaseController(base_speed=10.0, verbose=True)

    # Setup keyboard events
    appwindow = omni.appwindow.get_default_app_window()
    if appwindow and base_controller.setup_keyboard(appwindow):
        print(f"         ✓ Keyboard control enabled")
    else:
        print(f"         ⚠ Keyboard control unavailable (headless mode)")

    # Step 7: Find the correct paths to the wheel joints
    print(f"\n[Step 7] Finding wheel joint paths...")
    wheel_joint_paths = []

    # Search for joints under the robot prim
    def find_joints(prim, joint_names, found_paths, depth=0, max_depth=5):
        if depth > max_depth:
            return
        for child in prim.GetChildren():
            child_name = child.GetName()
            if child_name in joint_names:
                found_paths.append(child.GetPath().pathString)
                print(f"         ✓ Found {child_name} at {child.GetPath()}")
            find_joints(child, joint_names, found_paths, depth + 1, max_depth)

    target_joints = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
    find_joints(robot_prim, target_joints, wheel_joint_paths)

    if len(wheel_joint_paths) != 3:
        print(f"         ✗ WARNING: Found {len(wheel_joint_paths)}/3 wheel joints")

    # Step 8: Setup PhysX drive parameters for wheel joints (BEFORE starting simulation)
    print(f"\n[Step 8] Configuring PhysX drives for wheel joints...")
    drives_configured = 0
    for joint_path in wheel_joint_paths:
        joint_prim = stage.GetPrimAtPath(joint_path)
        joint_name = joint_path.split('/')[-1]
        if joint_prim and joint_prim.IsValid():
            print(f"         - Configuring {joint_name}...")

            # Apply PhysX drive API if not present
            if not joint_prim.HasAPI(UsdPhysics.DriveAPI):
                UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

            # Ensure the joint has PhysX RevoluteJoint API
            if not joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(joint_prim)
                print(f"           - Applied PhysxJointAPI")

            # Get the drive
            drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
            if drive:
                # CRITICAL: Set drive type to "force" for velocity control
                drive.CreateTypeAttr().Set("force")

                # Set drive parameters for velocity control
                drive.GetDampingAttr().Set(1000.0)  # Damping for stability
                drive.GetStiffnessAttr().Set(0.0)   # Zero stiffness for velocity control
                drive.GetMaxForceAttr().Set(10000.0)  # Max force limit

                # CRITICAL: Initialize target velocity to 0
                drive.CreateTargetVelocityAttr().Set(0.0)

                # Enable the drive (make sure it's active)
                if hasattr(drive, 'CreateEnabledAttr'):
                    drive.CreateEnabledAttr().Set(True)
                    print(f"           - Drive enabled=True")

                drives_configured += 1
                print(f"           ✓ Drive configured (type=force, damping=1000, stiffness=0, max_force=10000)")
            else:
                print(f"           ✗ Failed to get drive API")
        else:
            print(f"         ✗ Joint not found: {joint_path}")

    print(f"         ✓ {drives_configured}/3 PhysX drives configured")

    # Step 9: Ensure physics scene exists
    print(f"\n[Step 9] Checking physics scene...")
    physics_scene_path = "/World/physicsScene"
    physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)
    if not physics_scene_prim or not physics_scene_prim.IsValid():
        print(f"         - Creating physics scene at {physics_scene_path}...")
        UsdPhysics.Scene.Define(stage, physics_scene_path)
        print(f"         ✓ Physics scene created")
    else:
        print(f"         ✓ Physics scene already exists")

    # Step 10: Start timeline to enable PhysX simulation
    print(f"\n[Step 10] Starting physics simulation...")
    timeline = omni.timeline.get_timeline_interface()

    # Play the timeline to activate PhysX
    timeline.play()
    print(f"         ✓ Timeline playing (PhysX active)")

    # Get physics dt
    physx_interface = omni.physx.get_physx_interface()
    dt = 1.0 / 60.0  # 60 Hz physics

    # Simulation parameters
    frame = 0

    print(f"         Physics timestep: {dt*1000:.1f}ms ({1/dt:.0f} Hz)")

    print(f"\n[Step 11] Starting simulation loop\n")

    # Store joint prims and drives for faster access
    joint_prims = []
    joint_drives = []
    for joint_path in wheel_joint_paths:
        jp = stage.GetPrimAtPath(joint_path)
        joint_prims.append(jp)
        if jp and jp.IsValid():
            joint_drives.append(UsdPhysics.DriveAPI.Get(jp, "angular"))
        else:
            joint_drives.append(None)

    # Track previous joint positions to detect actual movement
    prev_joint_positions = [None, None, None]
    movement_detected_count = 0

    print(f"\n         DETAILED DIAGNOSTICS ENABLED")
    print(f"         - Logging every frame when movement commanded")
    print(f"         - Tracking joint positions, velocities, and drive state\n")

    try:
        while simulation_app.is_running():
            # Get wheel velocities from keyboard control
            wheel_velocities = base_controller.get_wheel_velocities()

            # Check if movement is commanded
            is_moving_cmd = any(abs(v) > 0.01 for v in wheel_velocities)

            # Apply velocities to robot wheels using USD DriveAPI
            velocities_applied = 0
            for i in range(len(wheel_joint_paths)):
                if i < len(wheel_velocities) and joint_drives[i]:
                    # Set target velocity in radians/sec
                    target_vel_attr = joint_drives[i].GetTargetVelocityAttr()
                    if target_vel_attr:
                        target_vel_attr.Set(float(wheel_velocities[i]))
                        velocities_applied += 1

            # Update rendering (timeline handles physics stepping)
            simulation_app.update()

            # Increment frame counter
            frame += 1

            # COMPREHENSIVE DIAGNOSTICS - Log every 10 frames when moving
            if is_moving_cmd and frame % 10 == 0:
                sim_time = timeline.get_current_time()
                is_playing = timeline.is_playing()

                print(f"\n[Frame {frame:4d}] Time: {sim_time:.3f}s | Timeline: {'PLAYING' if is_playing else 'STOPPED'}")
                print(f"  Command: [{wheel_velocities[0]:6.2f}, {wheel_velocities[1]:6.2f}, {wheel_velocities[2]:6.2f}] rad/s")
                print(f"  Applied: {velocities_applied}/3 drives")

                # Read actual joint states
                actual_movement = False
                for i, (joint_prim, joint_path) in enumerate(zip(joint_prims, wheel_joint_paths)):
                    if joint_prim and joint_prim.IsValid():
                        joint_name = joint_path.split('/')[-1]

                        # Read PhysX state attributes
                        vel_attr = joint_prim.GetAttribute("physxJoint:jointVelocity")
                        pos_attr = joint_prim.GetAttribute("physxJoint:jointPosition")

                        actual_vel = vel_attr.Get() if vel_attr else None
                        actual_pos = pos_attr.Get() if pos_attr else None

                        # Get drive target
                        drive = joint_drives[i]
                        target_vel = drive.GetTargetVelocityAttr().Get() if drive else None

                        # Check if joint position changed (actual movement)
                        pos_changed = "?"
                        if actual_pos is not None and prev_joint_positions[i] is not None:
                            pos_delta = actual_pos - prev_joint_positions[i]
                            if abs(pos_delta) > 0.001:
                                pos_changed = f"+{pos_delta:.4f}"
                                actual_movement = True
                            else:
                                pos_changed = "STATIC"

                        prev_joint_positions[i] = actual_pos

                        # Format values
                        target_str = f"{target_vel:7.2f}" if target_vel is not None else "   N/A"
                        vel_str = f"{actual_vel:7.2f}" if actual_vel is not None else "   N/A"
                        pos_str = f"{actual_pos:7.3f}" if actual_pos is not None else "   N/A"

                        print(f"  {joint_name:15s}: tgt={target_str} | vel={vel_str} | pos={pos_str} | Δ={pos_changed}")

                if actual_movement:
                    movement_detected_count += 1
                    print(f"  ✓ MOVEMENT DETECTED (count: {movement_detected_count})")
                else:
                    print(f"  ✗ NO MOVEMENT (robot may be stuck or drives not active)")

            # Regular status updates
            if frame % 120 == 0:
                sim_time = timeline.get_current_time()
                print(f"\n[Status] Frame {frame}, Sim Time: {sim_time:.2f}s, Movement count: {movement_detected_count}")

    except KeyboardInterrupt:
        print("\n         Interrupted by user")

    print("\n[Shutdown] Closing...")
    base_controller.cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
        print("Simulation closed.\n")
