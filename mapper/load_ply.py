import open3d as o3d
import numpy as np

ply_path = "mapper\points.ply"

# Load
pcd = o3d.io.read_point_cloud(ply_path)

# Info
print("=" * 60)
print(f"PLY File: {ply_path}")
print("=" * 60)
print(f"Number of points: {len(pcd.points)}")
print(f"Has colors: {pcd.has_colors()}")
print(f"Has normals: {pcd.has_normals()}")

if len(pcd.points) > 0:
    points = np.asarray(pcd.points)
    print(f"\nBounding box:")
    print(f"  Min: {points.min(axis=0)}")
    print(f"  Max: {points.max(axis=0)}")
    print(f"  Center: {points.mean(axis=0)}")
    print(f"  Size: {points.max(axis=0) - points.min(axis=0)}")

    # Create a planar surface centered at x=0, y=0 (z = 0) and sized to cover the point cloud
    # The plane is a simple rectangle (two triangles) and is painted light gray.
    min_xy = points.min(axis=0)[:2]
    max_xy = points.max(axis=0)[:2]
    extent_xy = max_xy - min_xy
    # Ensure a minimum visible plane size
    min_size = 1.0
    size_x = max(min_size, extent_xy[0] * 1.5)
    size_y = max(min_size, extent_xy[1] * 1.5)

    # Plane will be centered at (0,0,0) in X/Y; set z plane height to 0.0
    plane_z = 0.0

    verts = np.array([
        [-size_x / 2.0, -size_y / 2.0, plane_z],
        [ size_x / 2.0, -size_y / 2.0, plane_z],
        [ size_x / 2.0,  size_y / 2.0, plane_z],
        [-size_x / 2.0,  size_y / 2.0, plane_z],
    ], dtype=float)
    tris = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)

    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(verts)
    plane.triangles = o3d.utility.Vector3iVector(tris)
    plane.compute_vertex_normals()
    # Change plane color to a light blue for better visibility
    plane.paint_uniform_color([0.2, 0.6, 0.9])

    # Save original geometry so we can fully reset position and orientation later
    _orig_plane_verts = np.asarray(plane.vertices).copy()
    _orig_plane_tris = np.asarray(plane.triangles).copy()

    print(f"\nAdded plane at x=0,y=0,z={plane_z} with size_x={size_x:.2f}, size_y={size_y:.2f}")

# Visualize
print("\nPress 'Q' to close viewer...")

# Visualize both the point cloud and the plane (if created). The plane variable exists only
# when a point cloud was loaded and the plane was created above.
geometries = [pcd]
if 'plane' in locals():
    geometries.append(plane)


def _make_callbacks(mesh, trans_step, rot_step_rad):
    """Return a dict mapping key codes to callback functions for interactive control.

    Controls:
      I/K: move +Y / -Y
      J/L: move -X / +X
      U/O: move +Z / -Z
      Q/E: rotate CCW / CW around Z (degrees)
      R  : reset plane to origin and zero rotation
      H  : print help
    """
    import math

    # store initial geometry and transform information for reset
    init_center = mesh.get_center()
    orig_verts = None
    try:
        orig_verts = mesh.vertices
    except Exception:
        orig_verts = None

    def _update(vis):
        try:
            mesh.compute_vertex_normals()
        except Exception:
            pass
        vis.update_geometry(mesh)
        return False

    def move(dx, dy, dz):
        def _cb(vis):
            mesh.translate((dx, dy, dz), relative=True)
            return _update(vis)
        return _cb

    # We'll maintain mutable state for pivot mode (center vs world origin), rotation step,
    # and a cumulative rotation matrix for status reporting.
    state = {
        'pivot_is_center': True,
        'rot_step_deg': np.rad2deg(rot_step_rad),
        'R': np.eye(3, dtype=float),
    }

    def _get_pivot():
        return mesh.get_center() if state['pivot_is_center'] else (0.0, 0.0, 0.0)

    def rotate_axis(axis: np.ndarray, angle_rad: float):
        def _cb(vis):
            center = _get_pivot()
            # Build rotation matrix using Rodrigues' formula for arbitrary axis
            axis_n = axis / np.linalg.norm(axis)
            ux, uy, uz = axis_n
            c = math.cos(angle_rad)
            s = math.sin(angle_rad)
            R = np.array([
                [c + ux * ux * (1 - c),     ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
                [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c),     uy * uz * (1 - c) - ux * s],
                [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
            ])
            mesh.rotate(R, center=center)
            # Update cumulative rotation (R_new = R * R_old)
            state['R'] = R @ state['R']
            print(f"Rotated around axis {axis} by {np.rad2deg(angle_rad):.1f} deg. New Euler (Z,Y,X): { _euler_from_R(state['R']) }")
            return _update(vis)
        return _cb

    def rotate_axis_dynamic(axis: np.ndarray, angle_deg: float):
        # angle_deg can be positive or negative; uses state['rot_step_deg'] if angle_deg==None
        def _cb(vis):
            a = angle_deg if angle_deg is not None else state['rot_step_deg']
            return rotate_axis(axis, np.deg2rad(a))(vis)
        return _cb

    def _euler_from_R(R: np.ndarray) -> tuple[float, float, float]:
        # Returns yaw(Z), pitch(Y), roll(X) in degrees using ZYX (yaw-pitch-roll)
        sy = -R[2, 0]
        cy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.atan2(sy, cy)
        roll = math.atan2(R[2, 1], R[2, 2])
        return (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll))

    def reset_cb(vis):
        # Restore original vertex positions exactly (undo rotations and translations)
        try:
            # orig verts were stored globally when plane was created
            global _orig_plane_verts, _orig_plane_tris
            mesh.vertices = o3d.utility.Vector3dVector(_orig_plane_verts.copy())
            mesh.triangles = o3d.utility.Vector3iVector(_orig_plane_tris.copy())
            mesh.compute_vertex_normals()
            # reset cumulative rotation
            state['R'] = np.eye(3, dtype=float)
        except Exception:
            # Fallback: translate to original center
            cur_center = mesh.get_center()
            mesh.translate((-cur_center[0], -cur_center[1], -cur_center[2]), relative=True)
            mesh.translate((init_center[0], init_center[1], init_center[2]), relative=True)
        return _update(vis)

    def _print_status():
        yaw, pitch, roll = _euler_from_R(state['R'])
        print(f"\nPlane center: {mesh.get_center()}")
        print(f"Pivot mode: {'center' if state['pivot_is_center'] else 'world origin (0,0,0)'}")
        print(f"Rotation step (deg): {state['rot_step_deg']:.1f}")
        print(f"Cumulative Euler (yaw, pitch, roll) in deg: ({yaw:.1f}, {pitch:.1f}, {roll:.1f})")

    def help_cb(vis):
        print("\nInteractive plane controls:")
        print("  I/K : move +Y / -Y")
        print("  J/L : move -X / +X")
        print("  U/O : move +Z / -Z")
        print("  Q/E : yaw +/- around Z (same as before)")
        print("  Z/X : pitch +/- around X")
        print("  C/V : roll +/- around Y")
        print("  [   : decrease rotation step by 1 deg")
        print("  ]   : increase rotation step by 1 deg")
        print("  P   : toggle pivot (center <-> world origin)")
        print("  R   : exact reset (restore original vertices)")
        print("  H   : show this help")
        _print_status()
        return False

    return {
        ord('I'): move(0.0, trans_step, 0.0),
        ord('K'): move(0.0, -trans_step, 0.0),
        ord('J'): move(-trans_step, 0.0, 0.0),
        ord('L'): move(trans_step, 0.0, 0.0),
        ord('U'): move(0.0, 0.0, trans_step),
        ord('O'): move(0.0, 0.0, -trans_step),
        # yaw (around Z) - dynamic step
        ord('Q'): rotate_axis_dynamic(np.array([0.0, 0.0, 1.0]), None),
        ord('q'): rotate_axis_dynamic(np.array([0.0, 0.0, 1.0]), None),
        ord('E'): rotate_axis_dynamic(np.array([0.0, 0.0, 1.0]), None),
        ord('e'): rotate_axis_dynamic(np.array([0.0, 0.0, 1.0]), None),
        # pitch (around X)
        ord('Z'): rotate_axis_dynamic(np.array([1.0, 0.0, 0.0]), None),
        ord('z'): rotate_axis_dynamic(np.array([1.0, 0.0, 0.0]), None),
        ord('X'): rotate_axis_dynamic(np.array([1.0, 0.0, 0.0]), -state['rot_step_deg']),
        ord('x'): rotate_axis_dynamic(np.array([1.0, 0.0, 0.0]), -state['rot_step_deg']),
        # roll (around Y)
        ord('C'): rotate_axis_dynamic(np.array([0.0, 1.0, 0.0]), None),
        ord('c'): rotate_axis_dynamic(np.array([0.0, 1.0, 0.0]), None),
        ord('V'): rotate_axis_dynamic(np.array([0.0, 1.0, 0.0]), -state['rot_step_deg']),
        ord('v'): rotate_axis_dynamic(np.array([0.0, 1.0, 0.0]), -state['rot_step_deg']),
        # large-step yaw
        ord('G'): rotate_axis_dynamic(np.array([0.0, 0.0, 1.0]), 15.0),
        ord('g'): rotate_axis_dynamic(np.array([0.0, 0.0, 1.0]), 15.0),
        ord('T'): rotate_axis_dynamic(np.array([0.0, 0.0, 1.0]), -15.0),
        ord('t'): rotate_axis_dynamic(np.array([0.0, 0.0, 1.0]), -15.0),
        # decrease/increase rotation step
    ord('['): (lambda vis: (state.update({'rot_step_deg': max(1.0, state['rot_step_deg'] - 1.0)}), print(f"Rotation step now {state['rot_step_deg']:.1f} deg"), False)[2]),
    ord(']'): (lambda vis: (state.update({'rot_step_deg': min(45.0, state['rot_step_deg'] + 1.0)}), print(f"Rotation step now {state['rot_step_deg']:.1f} deg"), False)[2]),
        # toggle pivot
    ord('P'): (lambda vis: (state.update({'pivot_is_center': not state['pivot_is_center']}), print(f"Pivot now: {'center' if state['pivot_is_center'] else 'world origin'}"), False)[2]),
        ord('R'): reset_cb,
        ord('H'): help_cb,
    }


if 'plane' in locals():
    # translation step proportional to plane size, rotation step fixed
    trans_step = max(size_x, size_y) * 0.05
    rot_step_rad = np.deg2rad(5.0)
    callbacks = _make_callbacks(plane, trans_step, rot_step_rad)
    print("Interactive controls available. Press 'H' in console for help.")

    # Start a lightweight Tkinter control panel on a background thread to show pose and
    # provide buttons. This is a simple fallback UI that runs alongside the Open3D viewer.
    def _start_tk_panel(mesh, state):
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception:
            print("Tkinter not available - skipping control panel")
            return

        root = tk.Tk()
        root.title("Plane Control Panel")

        lbl = ttk.Label(root, text="Plane pose:\n(loading...)", justify=tk.LEFT)
        lbl.grid(row=0, column=0, columnspan=3, padx=8, pady=8)

        def update_label():
            try:
                c = mesh.get_center()
                # try to get cumulative euler if state has R
                yaw = pitch = roll = 0.0
                if 'R' in state:
                    try:
                        yaw, pitch, roll = _euler_from_R(state['R'])
                    except Exception:
                        pass
                lbl.config(text=f"Plane pose:\npos=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})\nrot(yaw,pitch,roll)=({yaw:.1f},{pitch:.1f},{roll:.1f})\npivot={'center' if state['pivot_is_center'] else 'world'}\nrot_step={state['rot_step_deg']:.1f}deg")
            except Exception:
                pass
            root.after(200, update_label)

        def on_reset():
            try:
                global _orig_plane_verts, _orig_plane_tris
                mesh.vertices = o3d.utility.Vector3dVector(_orig_plane_verts.copy())
                mesh.triangles = o3d.utility.Vector3iVector(_orig_plane_tris.copy())
                mesh.compute_vertex_normals()
                if 'R' in state:
                    state['R'] = np.eye(3, dtype=float)
            except Exception:
                pass

        def on_toggle_pivot():
            state['pivot_is_center'] = not state['pivot_is_center']

        def on_inc_step():
            state['rot_step_deg'] = min(45.0, state['rot_step_deg'] + 1.0)

        def on_dec_step():
            state['rot_step_deg'] = max(1.0, state['rot_step_deg'] - 1.0)

        btn_reset = ttk.Button(root, text="Reset Plane", command=on_reset)
        btn_reset.grid(row=1, column=0, padx=4, pady=4)
        btn_pivot = ttk.Button(root, text="Toggle Pivot", command=on_toggle_pivot)
        btn_pivot.grid(row=1, column=1, padx=4, pady=4)
        btn_close = ttk.Button(root, text="Close", command=root.destroy)
        btn_close.grid(row=1, column=2, padx=4, pady=4)

        btn_dec = ttk.Button(root, text="- Step", command=on_dec_step)
        btn_dec.grid(row=2, column=0, padx=4, pady=4)
        btn_inc = ttk.Button(root, text="+ Step", command=on_inc_step)
        btn_inc.grid(row=2, column=1, padx=4, pady=4)

        update_label()
        root.mainloop()

    # shared state for panel and key callbacks
    ui_state = {'pivot_is_center': True, 'rot_step_deg': np.rad2deg(rot_step_rad), 'R': np.eye(3, dtype=float)}
    try:
        import threading
        panel_thread = threading.Thread(target=_start_tk_panel, args=(plane, ui_state), daemon=True)
        panel_thread.start()
    except Exception:
        print("Failed to start Tkinter control panel")

    # Merge state into keyboard callbacks by setting values after callback creation
    # (the callbacks keep their own state dict, but we want the GUI to reflect changes too)
    # For simplicity leave them separate; both modify the mesh directly.

    o3d.visualization.draw_geometries_with_key_callbacks(geometries, callbacks)
else:
    o3d.visualization.draw_geometries(geometries,
                                      window_name="COLMAP Points",
                                      width=1024,
                                      height=768)