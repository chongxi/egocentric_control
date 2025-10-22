import argparse
import heapq
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - Matplotlib optional
    plt = None


GridIndex = Tuple[int, int]
WorldPoint = Tuple[float, float]


@dataclass
class RouteResult:
    start_world: WorldPoint
    goal_world: WorldPoint
    path_world: List[WorldPoint]
    path_length_m: float


class MapRouter:
    def __init__(self, map_path: Path):
        self.map_path = Path(map_path)
        if self.map_path.suffix.lower() != ".pgm":
            raise ValueError("Expected a .pgm map file.")

        self.yaml_path = self.map_path.with_suffix(".yaml")
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Matching YAML file not found: {self.yaml_path}")

        self.metadata = self._load_metadata()
        self.resolution = float(self.metadata["resolution"])
        self.origin = tuple(self.metadata["origin"])  # type: ignore[arg-type]
        self.occupied_thresh = float(self.metadata.get("occupied_thresh", 0.65))
        self.free_thresh = float(self.metadata.get("free_thresh", 0.196))
        self.negate = bool(self.metadata.get("negate", 0))

        self.map_data = self._load_map()
        self.height, self.width = self.map_data.shape

        # Robot physical dimensions (width x depth in meters)
        self.robot_width = 0.3  # meters
        self.robot_depth = 0.2  # meters

        self.normalized_map = self._normalize_map()
        self.walkable = self._compute_walkable_mask()

        # Create inflated walkable mask accounting for robot size
        self.walkable_inflated = self._inflate_obstacles()

        self._points3d, self._colors3d = self._create_point_cloud_data()
        self.map_center = self._points3d.mean(axis=0)
        self.min_bounds = self._points3d.min(axis=0)
        self.max_bounds = self._points3d.max(axis=0)

    def _load_metadata(self):
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_map(self):
        with open(self.map_path, "rb") as f:
            magic = f.readline().strip()
            if magic != b"P5":
                raise ValueError(f"Unsupported PGM format '{magic.decode()}' (expected P5).")

            def next_token():
                while True:
                    line = f.readline()
                    if not line:
                        raise ValueError("Unexpected end of file while reading PGM header.")
                    line = line.strip()
                    if not line or line.startswith(b"#"):
                        continue
                    return line

            dims = next_token().split()
            if len(dims) != 2:
                dims.extend(next_token().split())
            width, height = map(int, dims)

            max_val = int(next_token())
            if max_val > 255:
                raise ValueError("Only 8-bit PGM files are supported.")

            data = f.read(width * height)
            if len(data) != width * height:
                raise ValueError("PGM file truncated or invalid.")

        return np.frombuffer(data, dtype=np.uint8).reshape((height, width))

    def _normalize_map(self):
        normalized = self.map_data.astype(np.float32) / 255.0
        if self.negate:
            normalized = 1.0 - normalized
        return normalized

    def _compute_walkable_mask(self):
        occupied = self.normalized_map >= self.occupied_thresh
        free = self.normalized_map <= self.free_thresh
        walkable = ~occupied & free
        return walkable

    def _inflate_obstacles(self):
        """Inflate obstacles by robot footprint to create safe navigation space."""
        from scipy.ndimage import binary_erosion

        # Calculate inflation radius in cells (use max of width/depth for safety)
        inflation_radius_m = max(self.robot_width, self.robot_depth) / 2.0
        inflation_cells = int(np.ceil(inflation_radius_m / self.resolution))

        # Create circular structuring element for inflation
        y, x = np.ogrid[-inflation_cells:inflation_cells+1, -inflation_cells:inflation_cells+1]
        structure = x**2 + y**2 <= inflation_cells**2

        # Erode the walkable mask (which inflates obstacles)
        inflated_walkable = binary_erosion(self.walkable, structure=structure)

        return inflated_walkable

    def world_to_grid(self, point: WorldPoint) -> GridIndex:
        x, y = point
        col_float = (x - self.origin[0]) / self.resolution
        row_float = self.height - 1 - (y - self.origin[1]) / self.resolution
        col = int(round(col_float))
        row = int(round(row_float))
        return row, col

    def grid_to_world(self, cell: GridIndex) -> WorldPoint:
        row, col = cell
        x = self.origin[0] + col * self.resolution
        y = self.origin[1] + (self.height - row - 1) * self.resolution
        return x, y

    def clamp_cell(self, cell: GridIndex) -> Optional[GridIndex]:
        row, col = cell
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def find_nearest_walkable(self, cell: GridIndex) -> GridIndex:
        clamped = self.clamp_cell(cell)
        if clamped is None:
            raise ValueError("Cell outside of map bounds.")
        # Use inflated walkable mask for safety
        if self.walkable_inflated[clamped]:
            return clamped

        visited = np.zeros_like(self.walkable_inflated, dtype=bool)
        queue = deque([clamped])
        visited[clamped] = True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    continue
                if visited[nr, nc]:
                    continue
                visited[nr, nc] = True
                # Use inflated walkable mask
                if self.walkable_inflated[nr, nc]:
                    return nr, nc
                queue.append((nr, nc))

        raise RuntimeError("No walkable cell found in the map (accounting for robot size).")

    def compute_default_start_goal(self) -> Tuple[WorldPoint, WorldPoint]:
        # Default start near map center
        center_cell = (self.height // 2, self.width // 2)
        start_cell = self.find_nearest_walkable(center_cell)

        if not self.walkable_inflated[start_cell]:
            raise RuntimeError("Unable to find a walkable start cell (accounting for robot size).")

        # Goal: farthest walkable cell reachable from start
        distances = np.full((self.height, self.width), -1, dtype=np.int32)
        queue = deque([start_cell])
        distances[start_cell] = 0

        while queue:
            current = queue.popleft()
            for neighbor, _ in self._neighbors(current):
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        max_distance = distances.max()
        if max_distance <= 0:
            raise RuntimeError("Walkable area too small to compute a default goal.")

        goal_candidates = np.argwhere(distances == max_distance)
        goal_cell = tuple(int(v) for v in goal_candidates[0])

        return self.grid_to_world(start_cell), self.grid_to_world(goal_cell)

    def _create_point_cloud_data(self):
        rows = np.arange(self.height)
        cols = np.arange(self.width)
        jj, ii = np.meshgrid(cols, rows)

        xs = self.origin[0] + jj * self.resolution
        ys = self.origin[1] + (self.height - ii - 1) * self.resolution
        zs = np.zeros_like(xs, dtype=np.float32)

        points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        colors = (1.0 - self.normalized_map).reshape(-1, 1)
        colors = np.repeat(colors, 3, axis=1)
        return points.astype(np.float32), colors.astype(np.float32)

    def _neighbors(self, cell: GridIndex) -> Iterable[Tuple[GridIndex, float]]:
        row, col = cell
        directions = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        for dr, dc, cost in directions:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                continue
            # Use inflated walkable mask for path planning
            if not self.walkable_inflated[nr, nc]:
                continue
            if abs(dr) == 1 and abs(dc) == 1:
                if not (self.walkable_inflated[row, nc] and self.walkable_inflated[nr, col]):
                    continue
            yield (nr, nc), cost

    @staticmethod
    def _heuristic(a: GridIndex, b: GridIndex) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _reconstruct_path(self, came_from: dict, current: GridIndex) -> List[GridIndex]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_path(
        self,
        start_world: WorldPoint,
        goal_world: WorldPoint,
        *,
        snap_points: bool = True,
    ) -> RouteResult:
        start_cell = self.world_to_grid(start_world)
        goal_cell = self.world_to_grid(goal_world)

        if snap_points:
            start_cell = self.find_nearest_walkable(start_cell)
            goal_cell = self.find_nearest_walkable(goal_cell)
        else:
            start_cell = self.clamp_cell(start_cell)
            goal_cell = self.clamp_cell(goal_cell)
            if start_cell is None or goal_cell is None:
                raise ValueError("Start or goal outside of map bounds.")
            # Check inflated walkable mask for path planning safety
            if not self.walkable_inflated[start_cell] or not self.walkable_inflated[goal_cell]:
                raise ValueError("Start or goal is not on a free cell (accounting for robot size).")

        open_heap: List[Tuple[float, int, GridIndex]] = []
        counter = 0

        g_score = np.full((self.height, self.width), np.inf, dtype=np.float32)
        g_score[start_cell] = 0.0

        f_start = self._heuristic(start_cell, goal_cell)
        heapq.heappush(open_heap, (f_start, counter, start_cell))

        came_from = {}
        in_open = np.zeros_like(self.walkable, dtype=bool)
        in_open[start_cell] = True

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal_cell:
                path_cells = self._reconstruct_path(came_from, current)
                path_world = [self.grid_to_world(cell) for cell in path_cells]
                path_length = self._path_length_world(path_world)
                return RouteResult(
                    start_world=self.grid_to_world(start_cell),
                    goal_world=self.grid_to_world(goal_cell),
                    path_world=path_world,
                    path_length_m=path_length,
                )

            in_open[current] = False

            for neighbor, move_cost in self._neighbors(current):
                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_cell)
                    if not in_open[neighbor]:
                        counter += 1
                        heapq.heappush(open_heap, (f_score, counter, neighbor))
                        in_open[neighbor] = True

        raise RuntimeError("Path not found between start and goal.")

    @staticmethod
    def _path_length_world(path: List[WorldPoint]) -> float:
        if len(path) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(path)):
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            length += math.hypot(x1 - x0, y1 - y0)
        return length


class InteractiveRouterUI:
    """Matplotlib-based interactive viewer that lets users drag the goal point."""

    def __init__(self, router: MapRouter, initial_route: RouteResult):
        if plt is None:
            raise RuntimeError("Matplotlib is required for interactive mode.")

        self.router = router
        self.current_route = initial_route
        self.start_cell = self.router.world_to_grid(initial_route.start_world)
        if not self.router.walkable[self.start_cell]:
            raise RuntimeError("Interactive mode requires a walkable start cell.")
        self.current_goal_cell = self.router.world_to_grid(initial_route.goal_world)

        self.dragging = False
        self.drag_threshold = max(self.router.resolution * 4.0, 0.1)

        # Playback state
        self.robot_speed = 0.2  # m/s
        self.is_playing = False
        self.robot_position = initial_route.start_world  # Current robot position
        self.path_progress = 0.0  # Distance traveled along path
        self.animation_timer = None

        display_map = np.flipud(1.0 - self.router.normalized_map)
        self.map_extent = [
            self.router.origin[0],
            self.router.origin[0] + self.router.width * self.router.resolution,
            self.router.origin[1],
            self.router.origin[1] + self.router.height * self.router.resolution,
        ]

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(
            display_map,
            cmap="gray",
            origin="lower",
            extent=self.map_extent,
            vmin=0.0,
            vmax=1.0,
        )
        self.ax.set_xlim(self.map_extent[0], self.map_extent[1])
        self.ax.set_ylim(self.map_extent[2], self.map_extent[3])
        self.ax.set_aspect("equal", adjustable="box")

        self.path_line, = self.ax.plot([], [], color="red", linewidth=2.5, zorder=2)
        self.goal_artist = self.ax.scatter(
            initial_route.goal_world[0],
            initial_route.goal_world[1],
            c="royalblue",
            s=90,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
            label="Goal",
        )

        # Robot position as rectangle showing physical footprint
        from matplotlib.patches import Rectangle
        self.robot_rect = Rectangle(
            (self.robot_position[0] - self.router.robot_width / 2,
             self.robot_position[1] - self.router.robot_depth / 2),
            self.router.robot_width,
            self.router.robot_depth,
            angle=0,
            facecolor="orange",
            edgecolor="black",
            linewidth=1.5,
            zorder=4,
            label="Robot",
        )
        self.ax.add_patch(self.robot_rect)

        # Robot center marker (small triangle for orientation)
        self.robot_center_artist = self.ax.scatter(
            self.robot_position[0],
            self.robot_position[1],
            c="darkred",
            s=60,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

        self.ax.set_title("Interactive Path Planner - Drag goal or click Start to move robot")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.legend(loc="lower right", frameon=True)

        self.status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )

        # Add control buttons
        from matplotlib.widgets import Button

        # Adjust figure to make room for buttons
        self.fig.subplots_adjust(bottom=0.15)

        # Start button
        ax_start = self.fig.add_axes([0.2, 0.05, 0.15, 0.05])
        self.btn_start = Button(ax_start, 'Start')
        self.btn_start.on_clicked(self._on_start)

        # Stop button
        ax_stop = self.fig.add_axes([0.4, 0.05, 0.15, 0.05])
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_stop.on_clicked(self._on_stop)

        # Reset button
        ax_reset = self.fig.add_axes([0.6, 0.05, 0.15, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._set_status(self.current_route)
        self._update_plot(self.current_route)

    def run(self):
        plt.show()

    def _on_start(self, _event):
        """Start the robot movement playback."""
        if not self.is_playing and len(self.current_route.path_world) > 1:
            self.is_playing = True
            if self.animation_timer is None:
                # Start animation with ~30 FPS (33ms interval)
                self.animation_timer = self.fig.canvas.new_timer(interval=33)
                self.animation_timer.add_callback(self._update_animation)
            self.animation_timer.start()

    def _on_stop(self, _event):
        """Stop the robot movement playback."""
        if self.is_playing:
            self.is_playing = False
            if self.animation_timer is not None:
                self.animation_timer.stop()

    def _on_reset(self, _event):
        """Reset the robot to the start position."""
        self._on_stop(None)
        self.robot_position = self.current_route.start_world
        self.path_progress = 0.0
        # Update robot rectangle and center marker
        self.robot_rect.set_xy((
            self.robot_position[0] - self.router.robot_width / 2,
            self.robot_position[1] - self.router.robot_depth / 2
        ))
        self.robot_center_artist.set_offsets(np.array([self.robot_position]))
        self.fig.canvas.draw_idle()

    def _update_animation(self):
        """Update robot position along the path."""
        if not self.is_playing or len(self.current_route.path_world) < 2:
            return

        # Time step (33ms = 0.033s)
        dt = 0.033
        distance_to_move = self.robot_speed * dt
        path = self.current_route.path_world

        # Update progress
        self.path_progress += distance_to_move

        # Check if we've reached the end
        total_path_length = self.current_route.path_length_m
        if self.path_progress >= total_path_length:
            # Reached the end of the path
            self.robot_position = path[-1]
            self.robot_artist.set_offsets(np.array([self.robot_position]))
            self.fig.canvas.draw_idle()
            self._on_stop(None)
            return

        # Calculate new position along the path
        total_distance = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            segment_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

            if total_distance + segment_length >= self.path_progress:
                # Robot is on this segment
                remaining_in_segment = self.path_progress - total_distance
                if segment_length > 0:
                    t = remaining_in_segment / segment_length
                    new_x = p1[0] + t * (p2[0] - p1[0])
                    new_y = p1[1] + t * (p2[1] - p1[1])
                    self.robot_position = (new_x, new_y)
                break
            total_distance += segment_length

        # Update robot rectangle and center marker
        self.robot_rect.set_xy((
            self.robot_position[0] - self.router.robot_width / 2,
            self.robot_position[1] - self.router.robot_depth / 2
        ))
        self.robot_center_artist.set_offsets(np.array([self.robot_position]))
        self.fig.canvas.draw_idle()

    def _point_inside_map(self, x: float, y: float) -> bool:
        xmin, xmax = self.router.min_bounds[0], self.router.max_bounds[0]
        ymin, ymax = self.router.min_bounds[1], self.router.max_bounds[1]
        tol = self.router.resolution * 0.5
        return (xmin - tol) <= x <= (xmax + tol) and (ymin - tol) <= y <= (ymax + tol)

    def _on_press(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        goal = self.current_route.goal_world
        distance = math.hypot(event.xdata - goal[0], event.ydata - goal[1])
        self.dragging = distance <= self.drag_threshold
        self._update_goal(event.xdata, event.ydata)

    def _on_release(self, event):
        if event.button != 1:
            return
        self.dragging = False

    def _on_motion(self, event):
        if not self.dragging or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._update_goal(event.xdata, event.ydata)

    def _update_goal(self, x: float, y: float):
        if not self._point_inside_map(x, y):
            self._set_status("Outside map bounds.", is_error=True)
            return

        goal_cell = self.router.world_to_grid((x, y))
        clamped = self.router.clamp_cell(goal_cell)
        if clamped is None:
            self._set_status("Outside map bounds.", is_error=True)
            return

        try:
            snapped_goal = self.router.find_nearest_walkable(clamped)
        except (ValueError, RuntimeError):
            self._set_status("No reachable free cell near cursor.", is_error=True)
            return

        if snapped_goal == self.current_goal_cell:
            return

        goal_world = self.router.grid_to_world(snapped_goal)

        # Plan from current robot position instead of original start
        try:
            new_route = self.router.find_path(self.robot_position, goal_world, snap_points=False)
        except (RuntimeError, ValueError):
            self._set_status("No path found to that location.", is_error=True)
            return

        self.current_route = new_route
        self.current_goal_cell = snapped_goal

        # Keep robot at current position, reset path progress to 0
        self.path_progress = 0.0

        self._update_plot(new_route)
        self._set_status(new_route)

    def _update_plot(self, route: RouteResult):
        if route.path_world:
            xs, ys = zip(*route.path_world)
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

        self.goal_artist.set_offsets(np.array([route.goal_world]))

        # Update robot rectangle position
        self.robot_rect.set_xy((
            self.robot_position[0] - self.router.robot_width / 2,
            self.robot_position[1] - self.router.robot_depth / 2
        ))
        self.robot_center_artist.set_offsets(np.array([self.robot_position]))

        self.fig.canvas.draw_idle()

    def _set_status(self, route_or_message, is_error: bool = False):
        if isinstance(route_or_message, RouteResult):
            text = (
                f"Length: {route_or_message.path_length_m:.2f} m\n"
                f"Steps: {len(route_or_message.path_world)}"
            )
            self.status_text.set_color("white")
        else:
            text = str(route_or_message)
            self.status_text.set_color("yellow" if is_error else "white")

        self.status_text.set_text(text)
        self.fig.canvas.draw_idle()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and visualize A* routes on the occupancy grid."
    )
    parser.add_argument("command", nargs="?", default="run", help="Use 'run' to execute routing.")
    parser.add_argument(
        "--map",
        type=Path,
        default=None,
        help="Path to the .pgm occupancy map. Defaults to occupancy_map.pgm next to router.py.",
    )
    parser.add_argument(
        "--start",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Start point in map coordinates (meters).",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Goal point in map coordinates (meters).",
    )
    parser.add_argument(
        "--viewer",
        choices=("matplotlib", "none"),
        default="matplotlib",
        help="Viewer backend: 'matplotlib' for the 2D interactive view, or 'none' to skip visualization.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Compute path without launching any viewer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command.lower() != "run":
        raise SystemExit("Unknown command. Only 'run' is supported.")

    map_path = args.map
    if map_path is None:
        map_path = Path(__file__).resolve().with_name("occupancy_map.pgm")

    router = MapRouter(map_path)

    if args.start and args.goal:
        start = (args.start[0], args.start[1])
        goal = (args.goal[0], args.goal[1])
    else:
        start, goal = router.compute_default_start_goal()

    route = router.find_path(start, goal)

    viewer_choice = "none" if args.no_show else args.viewer

    if viewer_choice == "none":
        return

    if viewer_choice == "matplotlib":
        if plt is None:
            return
        ui = InteractiveRouterUI(router, route)
        ui.run()


if __name__ == "__main__":
    main()
