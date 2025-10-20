import open3d as o3d
import numpy as np

ply_path = "mapper\points5.ply"

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
    size_x = max(min_size, extent_xy[0] * 3)
    size_y = max(min_size, extent_xy[1] * 3)

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
        'trans_step': trans_step,
        'R': np.eye(3, dtype=float),
        'filter_min_dist': -1.0,  # minimum orthogonal distance from plane
        'filter_max_dist': 1.0,   # maximum orthogonal distance from plane
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
    # Reduce spacing: changed from 0.5 * em to 0.2 * em for tighter layout
    panel = o3d.visualization.gui.Vert(0.2 * em, o3d.visualization.gui.Margins(0.3 * em))

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

    panel.add_fixed(0.25 * em)
    panel.add_child(o3d.visualization.gui.Label("Rotation (deg):"))
    yaw_label = o3d.visualization.gui.Label("Yaw:   0.0")
    pitch_label = o3d.visualization.gui.Label("Pitch: 0.0")
    roll_label = o3d.visualization.gui.Label("Roll:  0.0")
    panel.add_child(yaw_label)
    panel.add_child(pitch_label)
    panel.add_child(roll_label)

    panel.add_fixed(0.25 * em)
    panel.add_child(o3d.visualization.gui.Label("Settings:"))
    pivot_label = o3d.visualization.gui.Label("Pivot: center")
    rot_step_label = o3d.visualization.gui.Label(f"Rot Step: {ui_state['rot_step_deg']:.1f}°")
    trans_step_label = o3d.visualization.gui.Label(f"Trans Step: {ui_state['trans_step']:.3f}")
    panel.add_child(pivot_label)
    panel.add_child(rot_step_label)
    panel.add_child(trans_step_label)

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
    panel.add_fixed(0.4 * em)

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

    def on_inc_trans_step():
        ui_state['trans_step'] = min(1.0, ui_state['trans_step'] * 1.5)
        trans_step_label.text = f"Trans Step: {ui_state['trans_step']:.3f}"
        window.post_redraw()

    def on_dec_trans_step():
        ui_state['trans_step'] = max(0.001, ui_state['trans_step'] / 1.5)
        trans_step_label.text = f"Trans Step: {ui_state['trans_step']:.3f}"
        window.post_redraw()

    reset_btn = o3d.visualization.gui.Button("Reset Plane")
    reset_btn.set_on_clicked(on_reset)
    panel.add_child(reset_btn)

    pivot_btn = o3d.visualization.gui.Button("Toggle Pivot")
    pivot_btn.set_on_clicked(on_toggle_pivot)
    panel.add_child(pivot_btn)

    step_horiz = o3d.visualization.gui.Horiz(0.2 * em)
    dec_btn = o3d.visualization.gui.Button("- Rot")
    dec_btn.set_on_clicked(on_dec_step)
    inc_btn = o3d.visualization.gui.Button("+ Rot")
    inc_btn.set_on_clicked(on_inc_step)
    step_horiz.add_child(dec_btn)
    step_horiz.add_child(inc_btn)
    panel.add_child(step_horiz)

    trans_step_horiz = o3d.visualization.gui.Horiz(0.2 * em)
    dec_trans_btn = o3d.visualization.gui.Button("- Trans")
    dec_trans_btn.set_on_clicked(on_dec_trans_step)
    inc_trans_btn = o3d.visualization.gui.Button("+ Trans")
    inc_trans_btn.set_on_clicked(on_inc_trans_step)
    trans_step_horiz.add_child(dec_trans_btn)
    trans_step_horiz.add_child(inc_trans_btn)
    panel.add_child(trans_step_horiz)

    # Translation controls
    panel.add_fixed(0.4 * em)
    panel.add_child(o3d.visualization.gui.Label("Translation Controls:"))

    # X axis controls
    x_horiz = o3d.visualization.gui.Horiz(0.2 * em)
    x_minus_btn = o3d.visualization.gui.Button("-X")
    x_plus_btn = o3d.visualization.gui.Button("+X")

    def on_x_minus():
        plane.translate((-ui_state['trans_step'], 0.0, 0.0), relative=True)
        update_scene_geometry()

    def on_x_plus():
        plane.translate((ui_state['trans_step'], 0.0, 0.0), relative=True)
        update_scene_geometry()

    x_minus_btn.set_on_clicked(on_x_minus)
    x_plus_btn.set_on_clicked(on_x_plus)
    x_horiz.add_child(x_minus_btn)
    x_horiz.add_child(x_plus_btn)
    panel.add_child(x_horiz)

    # Y axis controls
    y_horiz = o3d.visualization.gui.Horiz(0.2 * em)
    y_minus_btn = o3d.visualization.gui.Button("-Y")
    y_plus_btn = o3d.visualization.gui.Button("+Y")

    def on_y_minus():
        plane.translate((0.0, -ui_state['trans_step'], 0.0), relative=True)
        update_scene_geometry()

    def on_y_plus():
        plane.translate((0.0, ui_state['trans_step'], 0.0), relative=True)
        update_scene_geometry()

    y_minus_btn.set_on_clicked(on_y_minus)
    y_plus_btn.set_on_clicked(on_y_plus)
    y_horiz.add_child(y_minus_btn)
    y_horiz.add_child(y_plus_btn)
    panel.add_child(y_horiz)

    # Z axis controls
    z_horiz = o3d.visualization.gui.Horiz(0.2 * em)
    z_minus_btn = o3d.visualization.gui.Button("-Z")
    z_plus_btn = o3d.visualization.gui.Button("+Z")

    def on_z_minus():
        plane.translate((0.0, 0.0, -ui_state['trans_step']), relative=True)
        update_scene_geometry()

    def on_z_plus():
        plane.translate((0.0, 0.0, ui_state['trans_step']), relative=True)
        update_scene_geometry()

    z_minus_btn.set_on_clicked(on_z_minus)
    z_plus_btn.set_on_clicked(on_z_plus)
    z_horiz.add_child(z_minus_btn)
    z_horiz.add_child(z_plus_btn)
    panel.add_child(z_horiz)

    # Rotation controls
    panel.add_fixed(0.4 * em)
    panel.add_child(o3d.visualization.gui.Label("Rotation Controls:"))

    # Yaw controls
    yaw_horiz = o3d.visualization.gui.Horiz(0.2 * em)
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
    pitch_horiz = o3d.visualization.gui.Horiz(0.2 * em)
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
    roll_horiz = o3d.visualization.gui.Horiz(0.2 * em)
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

    # Projection controls
    panel.add_fixed(0.4 * em)
    panel.add_child(o3d.visualization.gui.Label("Projection:"))

    # Distance filter controls
    panel.add_fixed(0.25 * em)
    panel.add_child(o3d.visualization.gui.Label("Distance Filter (orthogonal):"))

    filter_range_label = o3d.visualization.gui.Label(
        f"Range: [{ui_state['filter_min_dist']:.3f}, {ui_state['filter_max_dist']:.3f}]"
    )
    panel.add_child(filter_range_label)

    # Min distance input
    min_dist_horiz = o3d.visualization.gui.Horiz(0.2 * em)
    min_dist_label = o3d.visualization.gui.Label("Min:")
    min_dist_input = o3d.visualization.gui.TextEdit()
    min_dist_input.text_value = str(ui_state['filter_min_dist'])
    min_dist_horiz.add_child(min_dist_label)
    min_dist_horiz.add_child(min_dist_input)
    panel.add_child(min_dist_horiz)

    # Max distance input
    max_dist_horiz = o3d.visualization.gui.Horiz(0.2 * em)
    max_dist_label = o3d.visualization.gui.Label("Max:")
    max_dist_input = o3d.visualization.gui.TextEdit()
    max_dist_input.text_value = str(ui_state['filter_max_dist'])
    max_dist_horiz.add_child(max_dist_label)
    max_dist_horiz.add_child(max_dist_input)
    panel.add_child(max_dist_horiz)

    def on_update_filter():
        """Update filter range from text inputs."""
        try:
            min_val = float(min_dist_input.text_value)
            max_val = float(max_dist_input.text_value)

            if min_val > max_val:
                print(f"Warning: Min distance ({min_val}) > Max distance ({max_val}). Swapping values.")
                min_val, max_val = max_val, min_val
                min_dist_input.text_value = str(min_val)
                max_dist_input.text_value = str(max_val)

            ui_state['filter_min_dist'] = min_val
            ui_state['filter_max_dist'] = max_val
            filter_range_label.text = f"Range: [{min_val:.3f}, {max_val:.3f}]"
            print(f"Filter range updated: [{min_val:.3f}, {max_val:.3f}]")
            window.post_redraw()
        except ValueError:
            print(f"Error: Invalid number format. Please enter valid numbers.")
            print(f"  Min: '{min_dist_input.text_value}', Max: '{max_dist_input.text_value}'")

    update_filter_btn = o3d.visualization.gui.Button("Update Filter Range")
    update_filter_btn.set_on_clicked(on_update_filter)
    panel.add_child(update_filter_btn)

    panel.add_fixed(0.25 * em)

    def project_points_to_plane():
        """Project 3D point cloud onto the current plane surface to create 2D occupancy map."""
        import matplotlib.pyplot as plt
        from scipy.ndimage import binary_dilation

        # Get plane vertices to compute plane normal and basis vectors
        plane_verts = np.asarray(plane.vertices)

        # Compute plane coordinate system
        # Use two edges of the plane as basis vectors
        v0 = plane_verts[0]
        v1 = plane_verts[1]
        v3 = plane_verts[3]

        # Compute plane center
        plane_center = plane.get_center()

        # Basis vectors in plane
        x_axis = v1 - v0  # along plane X
        y_axis = v3 - v0  # along plane Y

        # Normalize
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Plane normal (Z axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Get point cloud points
        points_3d = np.asarray(pcd.points)

        # Project points onto plane coordinate system
        # Transform points to plane's local coordinate system (relative to center)
        points_relative = points_3d - plane_center

        # Calculate orthogonal distance from plane (distance along z_axis/normal)
        orthogonal_distances = np.dot(points_relative, z_axis)

        # Apply distance filter
        min_dist = ui_state['filter_min_dist']
        max_dist = ui_state['filter_max_dist']
        distance_mask = (orthogonal_distances >= min_dist) & (orthogonal_distances <= max_dist)

        # Filter points based on orthogonal distance
        points_relative_filtered = points_relative[distance_mask]
        points_3d_filtered = points_3d[distance_mask]

        print(f"\nDistance filtering:")
        print(f"  Filter range: [{min_dist:.3f}, {max_dist:.3f}]")
        print(f"  Total points: {len(points_3d)}")
        print(f"  Points after distance filter: {len(points_3d_filtered)} ({100 * len(points_3d_filtered) / len(points_3d):.1f}%)")

        # Project onto plane's 2D coordinate system
        points_x = np.dot(points_relative_filtered, x_axis)
        points_y = np.dot(points_relative_filtered, y_axis)

        # Compute plane dimensions
        plane_width = np.linalg.norm(v1 - v0)
        plane_height = np.linalg.norm(v3 - v0)

        # Create occupancy map with aspect ratio matching the plane
        # Use a high base resolution and scale to maintain aspect ratio
        # Increased from 800 to 4000 for much higher detail
        base_resolution = 4000
        aspect_ratio = plane_width / plane_height

        if aspect_ratio >= 1.0:
            # Wider than tall
            resolution_x = base_resolution
            resolution_y = int(base_resolution / aspect_ratio)
        else:
            # Taller than wide
            resolution_x = int(base_resolution * aspect_ratio)
            resolution_y = base_resolution

        occupancy_map = np.zeros((resolution_y, resolution_x), dtype=bool)

        # Map points to image coordinates
        # Normalize to [0, 1] range
        x_normalized = (points_x + plane_width / 2) / plane_width
        y_normalized = (points_y + plane_height / 2) / plane_height

        # Convert to pixel coordinates
        pixel_x = (x_normalized * resolution_x).astype(int)
        pixel_y = (y_normalized * resolution_y).astype(int)

        # Filter valid points (within plane bounds)
        valid_mask = (pixel_x >= 0) & (pixel_x < resolution_x) & (pixel_y >= 0) & (pixel_y < resolution_y)
        pixel_x = pixel_x[valid_mask]
        pixel_y = pixel_y[valid_mask]

        # Mark occupied cells
        occupancy_map[pixel_y, pixel_x] = True

        # Apply morphological dilation to make points more visible
        # Scale dilation iterations based on resolution for consistent appearance
        dilation_iterations = max(1, int(base_resolution / 1000))
        occupancy_map = binary_dilation(occupancy_map, iterations=dilation_iterations)
        print(f"  Applied {dilation_iterations} dilation iterations for point visibility")

        # Flip Y axis for proper display (image origin is top-left)
        occupancy_map = np.flipud(occupancy_map)

        print(f"\nProjection complete!")
        print(f"  Plane dimensions: {plane_width:.2f} x {plane_height:.2f} (aspect ratio: {aspect_ratio:.2f})")
        print(f"  Points within distance filter: {len(points_3d_filtered)}")
        print(f"  Projected points (within plane bounds): {valid_mask.sum()}")
        print(f"  Occupancy map resolution: {resolution_x}x{resolution_y} pixels")

        # Create a new matplotlib window to display the occupancy map
        from matplotlib.colors import ListedColormap
        import os
        from datetime import datetime

        # Calculate figure size based on resolution to maintain aspect ratio
        # Use a larger figure size for better display of high-resolution images
        fig_width = 16
        fig_height = fig_width * (resolution_y / resolution_x)
        fig_height = min(fig_height, 16)  # Cap height at 16 inches

        plt.figure(figsize=(fig_width, fig_height), dpi=100)
        ax = plt.gca()

        # Display occupancy map with custom colormap
        cmap = ListedColormap(['black', 'white'])
        im = ax.imshow(occupancy_map, cmap=cmap, origin='upper', interpolation='nearest')

        ax.set_title(f'2D Occupancy Map - High Resolution ({resolution_x}x{resolution_y})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X axis (pixels)', fontsize=12)
        ax.set_ylabel('Y axis (pixels)', fontsize=12)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Occupied', ticks=[0, 1])

        plt.tight_layout()

        # Save high-resolution image automatically
        output_dir = "mapper"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"occupancy_map_{resolution_x}x{resolution_y}_{timestamp}.png")

        # Save with high DPI for maximum quality
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"  High-resolution occupancy map saved to: {output_path}")

        plt.show(block=False)  # Non-blocking show

        print("  Close the matplotlib window when done viewing.")

        return occupancy_map

    def on_project():
        try:
            project_points_to_plane()
            print("Occupancy map generated successfully.")
        except Exception as e:
            print(f"Error during projection: {e}")
            import traceback
            traceback.print_exc()

    project_btn = o3d.visualization.gui.Button("Project to 2D Map")
    project_btn.set_on_clicked(on_project)
    panel.add_child(project_btn)

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
    # Fallback: simple visualization without plane
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Points",
                                      width=1024,
                                      height=768)