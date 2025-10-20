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
    print("Interactive controls available. Press 'H' in console for help.")

    # Create Open3D GUI application with integrated UI panels
    import math

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    window = app.create_window("Point Cloud Viewer with Plane Control", 1400, 900)

    # Shared state for plane control
    ui_state = {
        'pivot_is_center': True,
        'rot_step_deg': np.rad2deg(rot_step_rad),
        'R': np.eye(3, dtype=float),
    }

    # Create 3D scene widget
    scene = o3d.visualization.gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

    # Add geometries to scene
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    scene.scene.add_geometry("pcd", pcd, mat)

    mat_plane = o3d.visualization.rendering.MaterialRecord()
    mat_plane.shader = "defaultLit"
    scene.scene.add_geometry("plane", plane, mat_plane)

    # Setup camera
    bounds = scene.scene.bounding_box
    scene.setup_camera(60, bounds, bounds.get_center())

    # Create UI panel on the right side
    em = window.theme.font_size
    panel_width = 20 * em
    panel = o3d.visualization.gui.Vert(0.5 * em, o3d.visualization.gui.Margins(0.5 * em))

    # Title
    title_label = o3d.visualization.gui.Label("PLANE CONTROL PANEL")
    panel.add_child(title_label)

    # Pose information labels
    panel.add_child(o3d.visualization.gui.Label("Position:"))
    pos_x_label = o3d.visualization.gui.Label("X: 0.000")
    pos_y_label = o3d.visualization.gui.Label("Y: 0.000")
    pos_z_label = o3d.visualization.gui.Label("Z: 0.000")
    panel.add_child(pos_x_label)
    panel.add_child(pos_y_label)
    panel.add_child(pos_z_label)

    panel.add_fixed(0.5 * em)
    panel.add_child(o3d.visualization.gui.Label("Rotation (deg):"))
    yaw_label = o3d.visualization.gui.Label("Yaw:   0.0")
    pitch_label = o3d.visualization.gui.Label("Pitch: 0.0")
    roll_label = o3d.visualization.gui.Label("Roll:  0.0")
    panel.add_child(yaw_label)
    panel.add_child(pitch_label)
    panel.add_child(roll_label)

    panel.add_fixed(0.5 * em)
    panel.add_child(o3d.visualization.gui.Label("Settings:"))
    pivot_label = o3d.visualization.gui.Label("Pivot: center")
    rot_step_label = o3d.visualization.gui.Label(f"Rot Step: {ui_state['rot_step_deg']:.1f}°")
    panel.add_child(pivot_label)
    panel.add_child(rot_step_label)

    # Helper functions for transformations
    def _euler_from_R(R: np.ndarray) -> tuple:
        sy = -R[2, 0]
        cy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.atan2(sy, cy)
        roll = math.atan2(R[2, 1], R[2, 2])
        return (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll))

    def update_ui_labels():
        try:
            c = plane.get_center()
            pos_x_label.text = f"X: {c[0]:7.3f}"
            pos_y_label.text = f"Y: {c[1]:7.3f}"
            pos_z_label.text = f"Z: {c[2]:7.3f}"

            yaw, pitch, roll = _euler_from_R(ui_state['R'])
            yaw_label.text = f"Yaw:   {yaw:6.1f}"
            pitch_label.text = f"Pitch: {pitch:6.1f}"
            roll_label.text = f"Roll:  {roll:6.1f}"
        except Exception as e:
            print(f"Error updating labels: {e}")

    def update_scene_geometry():
        scene.scene.remove_geometry("plane")
        mat_plane = o3d.visualization.rendering.MaterialRecord()
        mat_plane.shader = "defaultLit"
        scene.scene.add_geometry("plane", plane, mat_plane)
        update_ui_labels()
        window.post_redraw()

    def rotate_plane(axis, angle_deg):
        center = plane.get_center() if ui_state['pivot_is_center'] else (0.0, 0.0, 0.0)
        angle_rad = np.deg2rad(angle_deg)
        axis_n = axis / np.linalg.norm(axis)
        ux, uy, uz = axis_n
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ])
        plane.rotate(R, center=center)
        ui_state['R'] = R @ ui_state['R']
        update_scene_geometry()

    # Buttons
    panel.add_fixed(em)

    def on_reset():
        global _orig_plane_verts, _orig_plane_tris
        plane.vertices = o3d.utility.Vector3dVector(_orig_plane_verts.copy())
        plane.triangles = o3d.utility.Vector3iVector(_orig_plane_tris.copy())
        plane.compute_vertex_normals()
        scene.scene.remove_geometry("plane")
        mat_plane = o3d.visualization.rendering.MaterialRecord()
        mat_plane.shader = "defaultLit"
        scene.scene.add_geometry("plane", plane, mat_plane)
        ui_state['R'] = np.eye(3, dtype=float)
        window.post_redraw()

    def on_toggle_pivot():
        ui_state['pivot_is_center'] = not ui_state['pivot_is_center']
        pivot_label.text = f"Pivot: {'center' if ui_state['pivot_is_center'] else 'world'}"
        window.post_redraw()

    def on_inc_step():
        ui_state['rot_step_deg'] = min(45.0, ui_state['rot_step_deg'] + 1.0)
        rot_step_label.text = f"Rot Step: {ui_state['rot_step_deg']:.1f}°"
        window.post_redraw()

    def on_dec_step():
        ui_state['rot_step_deg'] = max(1.0, ui_state['rot_step_deg'] - 1.0)
        rot_step_label.text = f"Rot Step: {ui_state['rot_step_deg']:.1f}°"
        window.post_redraw()

    reset_btn = o3d.visualization.gui.Button("Reset Plane")
    reset_btn.set_on_clicked(on_reset)
    panel.add_child(reset_btn)

    pivot_btn = o3d.visualization.gui.Button("Toggle Pivot")
    pivot_btn.set_on_clicked(on_toggle_pivot)
    panel.add_child(pivot_btn)

    step_horiz = o3d.visualization.gui.Horiz(0.5 * em)
    dec_btn = o3d.visualization.gui.Button("- Step")
    dec_btn.set_on_clicked(on_dec_step)
    inc_btn = o3d.visualization.gui.Button("+ Step")
    inc_btn.set_on_clicked(on_inc_step)
    step_horiz.add_child(dec_btn)
    step_horiz.add_child(inc_btn)
    panel.add_child(step_horiz)

    # Translation controls
    panel.add_fixed(em)
    panel.add_child(o3d.visualization.gui.Label("Translation Controls:"))

    # X axis controls
    x_horiz = o3d.visualization.gui.Horiz(0.5 * em)
    x_minus_btn = o3d.visualization.gui.Button("-X")
    x_plus_btn = o3d.visualization.gui.Button("+X")

    def on_x_minus():
        plane.translate((-trans_step, 0.0, 0.0), relative=True)
        update_scene_geometry()

    def on_x_plus():
        plane.translate((trans_step, 0.0, 0.0), relative=True)
        update_scene_geometry()

    x_minus_btn.set_on_clicked(on_x_minus)
    x_plus_btn.set_on_clicked(on_x_plus)
    x_horiz.add_child(x_minus_btn)
    x_horiz.add_child(x_plus_btn)
    panel.add_child(x_horiz)

    # Y axis controls
    y_horiz = o3d.visualization.gui.Horiz(0.5 * em)
    y_minus_btn = o3d.visualization.gui.Button("-Y")
    y_plus_btn = o3d.visualization.gui.Button("+Y")

    def on_y_minus():
        plane.translate((0.0, -trans_step, 0.0), relative=True)
        update_scene_geometry()

    def on_y_plus():
        plane.translate((0.0, trans_step, 0.0), relative=True)
        update_scene_geometry()

    y_minus_btn.set_on_clicked(on_y_minus)
    y_plus_btn.set_on_clicked(on_y_plus)
    y_horiz.add_child(y_minus_btn)
    y_horiz.add_child(y_plus_btn)
    panel.add_child(y_horiz)

    # Z axis controls
    z_horiz = o3d.visualization.gui.Horiz(0.5 * em)
    z_minus_btn = o3d.visualization.gui.Button("-Z")
    z_plus_btn = o3d.visualization.gui.Button("+Z")

    def on_z_minus():
        plane.translate((0.0, 0.0, -trans_step), relative=True)
        update_scene_geometry()

    def on_z_plus():
        plane.translate((0.0, 0.0, trans_step), relative=True)
        update_scene_geometry()

    z_minus_btn.set_on_clicked(on_z_minus)
    z_plus_btn.set_on_clicked(on_z_plus)
    z_horiz.add_child(z_minus_btn)
    z_horiz.add_child(z_plus_btn)
    panel.add_child(z_horiz)

    # Rotation controls
    panel.add_fixed(em)
    panel.add_child(o3d.visualization.gui.Label("Rotation Controls:"))

    # Yaw controls
    yaw_horiz = o3d.visualization.gui.Horiz(0.5 * em)
    yaw_minus_btn = o3d.visualization.gui.Button("-Yaw")
    yaw_plus_btn = o3d.visualization.gui.Button("+Yaw")

    def on_yaw_minus():
        rotate_plane(np.array([0.0, 0.0, 1.0]), -ui_state['rot_step_deg'])

    def on_yaw_plus():
        rotate_plane(np.array([0.0, 0.0, 1.0]), ui_state['rot_step_deg'])

    yaw_minus_btn.set_on_clicked(on_yaw_minus)
    yaw_plus_btn.set_on_clicked(on_yaw_plus)
    yaw_horiz.add_child(yaw_minus_btn)
    yaw_horiz.add_child(yaw_plus_btn)
    panel.add_child(yaw_horiz)

    # Pitch controls
    pitch_horiz = o3d.visualization.gui.Horiz(0.5 * em)
    pitch_minus_btn = o3d.visualization.gui.Button("-Pitch")
    pitch_plus_btn = o3d.visualization.gui.Button("+Pitch")

    def on_pitch_minus():
        rotate_plane(np.array([1.0, 0.0, 0.0]), -ui_state['rot_step_deg'])

    def on_pitch_plus():
        rotate_plane(np.array([1.0, 0.0, 0.0]), ui_state['rot_step_deg'])

    pitch_minus_btn.set_on_clicked(on_pitch_minus)
    pitch_plus_btn.set_on_clicked(on_pitch_plus)
    pitch_horiz.add_child(pitch_minus_btn)
    pitch_horiz.add_child(pitch_plus_btn)
    panel.add_child(pitch_horiz)

    # Roll controls
    roll_horiz = o3d.visualization.gui.Horiz(0.5 * em)
    roll_minus_btn = o3d.visualization.gui.Button("-Roll")
    roll_plus_btn = o3d.visualization.gui.Button("+Roll")

    def on_roll_minus():
        rotate_plane(np.array([0.0, 1.0, 0.0]), -ui_state['rot_step_deg'])

    def on_roll_plus():
        rotate_plane(np.array([0.0, 1.0, 0.0]), ui_state['rot_step_deg'])

    roll_minus_btn.set_on_clicked(on_roll_minus)
    roll_plus_btn.set_on_clicked(on_roll_plus)
    roll_horiz.add_child(roll_minus_btn)
    roll_horiz.add_child(roll_plus_btn)
    panel.add_child(roll_horiz)

    # Layout
    window.add_child(scene)
    window.add_child(panel)

    def on_layout(layout_context):
        r = window.content_rect
        scene.frame = r
        panel_rect = o3d.visualization.gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        panel.frame = panel_rect

    window.set_on_layout(on_layout)

    # Initialize UI with current values
    update_ui_labels()

    # Run the application
    app.run()

else:
    o3d.visualization.draw_geometries(geometries,
                                      window_name="COLMAP Points",
                                      width=1024,
                                      height=768)