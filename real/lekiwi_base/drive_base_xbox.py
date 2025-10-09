#!/usr/bin/env python
"""Drive the LeKiwi base with an Xbox controller left stick."""

import argparse
import math
from typing import Tuple

import pygame

from lerobot.robots.lekiwi_base import LeKiwiBaseConfig, LeKiwi_base

DPAD_BUTTONS = dict(left=14, right=13)  # adjust if needed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        default="/dev/tty.usbmodem5AB90691091",
        help="Serial port for the Feetech bus (e.g. /dev/ttyUSB0 on Linux or /dev/tty.usbmodemXXXX on macOS)",
    )
    parser.add_argument(
        "--no-handshake",
        action="store_true",
        help="Skip the Feetech handshake (useful while assigning motor IDs).",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip calibration even if no file is present.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=1.0,
        help="Maximum linear velocity (m/s) when the stick is fully deflected.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="Ignore stick input whose absolute value is below this threshold.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Command update rate (frames per second).",
    )
    parser.add_argument(
        "--yaw-speed",
        type=float,
        default=0.5,
        help="Yaw rate (rad/s) when D-pad left/right is held.",
    )
    return parser.parse_args()


def normalize_axis(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return max(-1.0, min(1.0, value))


def read_left_stick(js: pygame.joystick.Joystick) -> Tuple[float, float]:
    """
    Return (x, y) in [-1, 1].
    SDL on macOS maps axes 0/1 to the left stick; y is inverted so we flip it.
    """
    x = js.get_axis(0)
    y = -js.get_axis(1)
    return x, y


def read_dpad(js: pygame.joystick.Joystick) -> Tuple[int, int]:
    if js.get_numhats() > 0:
        return js.get_hat(0)

    def pressed(idx: int) -> bool:
        return js.get_button(idx) == 1

    hx = (-1 if pressed(DPAD_BUTTONS["left"]) else 0) + (1 if pressed(DPAD_BUTTONS["right"]) else 0)
    return hx, 0


def vector_to_velocities(
    x_axis: float,
    y_axis: float,
    max_speed: float,
) -> Tuple[float, float]:
    """
    Convert stick vector to field-centric x/y velocities.
    x_axis -> robot y (lateral), y_axis -> robot x (forward).
    """
    x_vel = y_axis * max_speed
    y_vel = x_axis * max_speed
    return x_vel, y_vel


def main() -> None:
    args = parse_args()

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise SystemExit("No controller found. Plug in/pair your Xbox controller and try again.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    config = LeKiwiBaseConfig(port=args.port)
    robot = LeKiwi_base(config)

    robot.connect(calibrate=not args.no_calibrate, handshake=not args.no_handshake)
    print("Connected:", robot.is_connected)
    print(f"Using controller: {joystick.get_name()}")
    print("Press Ctrl+C to stop.")

    clock = pygame.time.Clock()
    last_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    robot.send_action(last_action)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEREMOVED:
                    raise SystemExit("Controller disconnected.")

            x_axis_raw, y_axis_raw = read_left_stick(joystick)
            x_axis = normalize_axis(x_axis_raw, args.deadzone)
            y_axis = normalize_axis(y_axis_raw, args.deadzone)
            magnitude = math.hypot(x_axis, y_axis)

            if magnitude > 1.0:
                scale = 1.0 / magnitude
                x_axis *= scale
                y_axis *= scale

            x_vel, y_vel = vector_to_velocities(x_axis, y_axis, args.max_speed)
            hx, _ = read_dpad(joystick)
            if hx != 0:
                theta_vel = math.degrees(args.yaw_speed) * hx
            else:
                theta_vel = 0.0

            action = {"x.vel": x_vel, "y.vel": -y_vel, "theta.vel": theta_vel}

            if any(abs(action[k] - last_action[k]) > 1e-3 for k in action):
                robot.send_action(action)
                last_action = action

            clock.tick(args.fps)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.stop_base()
        robot.disconnect()
        pygame.quit()
        print("Disconnected.")


if __name__ == "__main__":
    main()
