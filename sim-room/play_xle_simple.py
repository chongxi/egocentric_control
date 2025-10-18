"""Simple approach: Load room_23.usd as base stage, then add robot as reference."""

import os
from isaacsim import SimulationApp

# Launch Isaac Sim with viewer
simulation_app = SimulationApp(launch_config={"headless": False})

import omni.kit.commands
import omni.usd
import omni.timeline
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf

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

    # Step 6: Start simulation
    print(f"\n[Step 6] Starting simulation loop")
    print(f"         Press Ctrl+C or close window to exit\n")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    frame = 0
    try:
        while simulation_app.is_running():
            simulation_app.update()
            frame += 1

            if frame % 120 == 0:
                current_time = timeline.get_current_time()
                print(f"         Frame {frame}, Time: {current_time:.2f}s")

    except KeyboardInterrupt:
        print("\n         Interrupted by user")

    print("\n[Shutdown] Pausing timeline and closing...")
    timeline.pause()


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
