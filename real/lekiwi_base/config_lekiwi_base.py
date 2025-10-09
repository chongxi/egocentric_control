"""Configuration for the LeKiwi base-only robot."""

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("lekiwi_base")
@dataclass
class LeKiwiBaseConfig(RobotConfig):
    """Configuration for the three-wheel LeKiwi base."""

    port: str = "/dev/tty.usbmodem5AB90691091"
    disable_torque_on_disconnect: bool = True

    # Geometry parameters used by the kinematics helpers.
    wheel_radius_m: float = 0.05
    base_radius_m: float = 0.125
    # Axis directions (degrees) for left, back, right wheels in the body frame.
    # wheel_axis_angles_deg: tuple[float, float, float] = (150.0, -90.0, 30.0)
    wheel_axis_angles_deg: tuple[float, float, float] = (150.0, -90.0, 30.0)

    # Max raw wheel command (ticks) before scaling.
    max_wheel_raw: int = 3000
