"""
XLE Robot Base Controller - Shared keyboard control for 6-DOF omniwheel movement.

Supports both Isaac Sim (direct) and Isaac Lab (framework) approaches.
"""

import numpy as np
import carb.input
from typing import Callable, Optional


class OmniBaseController:
    """
    Egocentric keyboard controller for 3-wheel omniwheel robot base.

    Supports 6 directions of movement:
    - Forward/Backward (W/S)
    - Strafe Left/Right (A/D)
    - Rotate CCW/CW (Q/E)
    """

    # Keyboard command mappings (egocentric control)
    KEYBOARD_MAPPINGS = {
        # Forward
        "W": [1.0, 0.0, 0.0],
        "UP": [1.0, 0.0, 0.0],
        "NUMPAD_8": [1.0, 0.0, 0.0],
        # Backward
        "S": [-1.0, 0.0, 0.0],
        "DOWN": [-1.0, 0.0, 0.0],
        "NUMPAD_2": [-1.0, 0.0, 0.0],
        # Strafe Left
        "A": [0.0, 1.0, 0.0],
        "LEFT": [0.0, 1.0, 0.0],
        "NUMPAD_4": [0.0, 1.0, 0.0],
        # Strafe Right
        "D": [0.0, -1.0, 0.0],
        "RIGHT": [0.0, -1.0, 0.0],
        "NUMPAD_6": [0.0, -1.0, 0.0],
        # Rotate Counter-Clockwise
        "Q": [0.0, 0.0, 1.0],
        "NUMPAD_7": [0.0, 0.0, 1.0],
        # Rotate Clockwise
        "E": [0.0, 0.0, -1.0],
        "NUMPAD_9": [0.0, 0.0, -1.0],
        # Stop
        "SPACE": 'stop',
        "NUMPAD_5": 'stop',
    }

    def __init__(self, base_speed: float = 10.0, verbose: bool = True):
        """
        Initialize the base controller.

        Args:
            base_speed: Base movement speed multiplier
            verbose: Enable debug output
        """
        self.base_speed = base_speed
        self.verbose = verbose

        # Command state
        self.base_command = np.zeros(3)  # [forward/back, left/right, rotate]
        self.pressed_keys = set()

        # Keyboard interface (initialized when setup_keyboard is called)
        self.input_interface = None
        self.keyboard = None
        self._subscription = None

    def setup_keyboard(self, appwindow) -> bool:
        """
        Setup keyboard event handler for Isaac Sim.

        Args:
            appwindow: Isaac Sim application window

        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.input_interface = carb.input.acquire_input_interface()
            self.keyboard = appwindow.get_keyboard()

            # Subscribe to keyboard events
            self._subscription = self.input_interface.subscribe_to_keyboard_events(
                self.keyboard,
                self._keyboard_event_handler
            )

            self.print_controls()
            return True

        except Exception as e:
            print(f"[ERROR] Failed to setup keyboard: {e}")
            return False

    def _keyboard_event_handler(self, event, *args, **kwargs) -> bool:
        """Internal keyboard event handler."""
        # Handle both string and object-based key events
        key_name = event.input.name if hasattr(event.input, "name") else event.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in self.KEYBOARD_MAPPINGS:
                # Ignore auto-repeat if key already held
                if key_name in self.pressed_keys:
                    return True
                self.pressed_keys.add(key_name)

                command = self.KEYBOARD_MAPPINGS[key_name]
                if command == 'stop':
                    self.base_command = np.zeros(3)
                else:
                    self.base_command += np.array(command)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in self.KEYBOARD_MAPPINGS:
                if key_name in self.pressed_keys:
                    self.pressed_keys.remove(key_name)
                command = self.KEYBOARD_MAPPINGS[key_name]
                if command != 'stop':
                    self.base_command -= np.array(command)

        return True

    def get_wheel_velocities(self, device=None) -> np.ndarray:
        """
        Convert base command to 3-wheel omniwheel velocities.

        Args:
            device: Optional torch device for tensor output

        Returns:
            Wheel velocities [wheel_0, wheel_1, wheel_2] as numpy array or torch tensor
        """
        import torch

        # Initialize velocities
        if device is not None:
            wheel_velocities = torch.zeros(3, dtype=torch.float32, device=device)
        else:
            wheel_velocities = np.zeros(3, dtype=np.float32)

        # Convert base command to wheel velocities
        # base_command: [forward/back, left/right, rotate]

        # Forward/backward component
        if self.base_command[0] != 0:
            if device is not None:
                wheel_velocities += torch.tensor(
                    [0, 1, -1], dtype=torch.float32, device=device
                ) * self.base_command[0] * self.base_speed * 1.5
            else:
                wheel_velocities += np.array([0, 1, -1]) * self.base_command[0] * self.base_speed * 1.5

        # Left/right component
        if self.base_command[1] != 0:
            if device is not None:
                wheel_velocities += torch.tensor(
                    [-1, 0.45, 0.45], dtype=torch.float32, device=device
                ) * self.base_command[1] * self.base_speed
            else:
                wheel_velocities += np.array([-1, 0.45, 0.45]) * self.base_command[1] * self.base_speed

        # Rotation component
        if self.base_command[2] != 0:
            if device is not None:
                wheel_velocities += torch.tensor(
                    [-1, -1, -1], dtype=torch.float32, device=device
                ) * self.base_command[2] * self.base_speed
            else:
                wheel_velocities += np.array([-1, -1, -1]) * self.base_command[2] * self.base_speed

        return wheel_velocities

    def is_moving(self) -> bool:
        """Check if base is currently commanded to move."""
        return not np.allclose(self.base_command, 0.0)

    def get_command(self) -> np.ndarray:
        """Get current base command [forward/back, left/right, rotate]."""
        return self.base_command.copy()

    def reset(self):
        """Reset command state."""
        self.base_command = np.zeros(3)
        self.pressed_keys.clear()
        if self.verbose:
            print("[INFO] Base controller reset")

    def cleanup(self):
        """Cleanup keyboard subscription."""
        if self._subscription is not None:
            self.input_interface.unsubscribe_to_keyboard_events(self.keyboard, self._subscription)
            self._subscription = None

    def print_controls(self):
        """Print keyboard control instructions."""
        print("\n" + "="*70)
        print("EGOCENTRIC KEYBOARD CONTROL - XLE OMNIWHEEL BASE")
        print("="*70)
        print("Movement Controls:")
        print("  W / ↑ / Numpad8     : Move Forward")
        print("  S / ↓ / Numpad2     : Move Backward")
        print("  A / ← / Numpad4     : Strafe Left")
        print("  D / → / Numpad6     : Strafe Right")
        print("")
        print("Rotation Controls:")
        print("  Q / Numpad7         : Rotate Counter-Clockwise")
        print("  E / Numpad9         : Rotate Clockwise")
        print("")
        print("Other:")
        print("  Space / Numpad5     : Stop All Movement")
        print("="*70 + "\n")

    def print_debug_info(self, frame: int, wheel_velocities=None):
        """
        Print debug information (call periodically, not every frame).

        Args:
            frame: Current frame number
            wheel_velocities: Optional wheel velocities to display
        """
        if not self.verbose or frame % 30 != 0:
            return

        if np.any(self.base_command != 0):
            cmd_str = f"[{self.base_command[0]:.2f}, {self.base_command[1]:.2f}, {self.base_command[2]:.2f}]"
            print(f"[Frame {frame}] Base command: {cmd_str}", end="")

            if wheel_velocities is not None:
                if hasattr(wheel_velocities, 'cpu'):  # torch tensor
                    vel = wheel_velocities.cpu().numpy()
                else:  # numpy array
                    vel = wheel_velocities
                vel_str = f"[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]"
                print(f" | Wheel vels: {vel_str}")
            else:
                print()


def print_headless_warning():
    """Print warning when running in headless mode."""
    print("\n" + "="*70)
    print("HEADLESS MODE DETECTED")
    print("="*70)
    print("Keyboard control is not available in headless mode.")
    print("The robot will remain stationary.")
    print("="*70 + "\n")
