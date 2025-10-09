#!/usr/bin/env python

"""LeKiwi base-only robot implementation."""

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots import Robot
from .config_lekiwi_base import LeKiwiBaseConfig

logger = logging.getLogger(__name__)


class LeKiwi_base(Robot):
    """Three-wheel LeKiwi base driven by Feetech motors."""

    config_class = LeKiwiBaseConfig
    name = "lekiwi_base"

    def __init__(self, config: LeKiwiBaseConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "base_back_wheel": Motor(0, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_left_wheel": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.base_motors = list(self.bus.motors.keys())

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(("x.vel", "y.vel", "theta.vel"), float)

    @property
    @cached_property
    def observation_features(self) -> dict[str, type]:
        return self._state_ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True, handshake: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect(handshake=handshake)
        if calibrate and not self.is_calibrated:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"Calibrating {self} base motors")

        homing_offsets = dict.fromkeys(self.base_motors, 0)
        range_mins = dict.fromkeys(self.base_motors, 0)
        range_maxes = dict.fromkeys(self.base_motors, 4095)

        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info("Calibration saved at %s", self.calibration_fpath)

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in self.base_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        self.bus.enable_torque()

    def setup_motors(self) -> None:
        for motor in self.base_motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_int = int(round(degps * steps_per_deg))
        return max(min(speed_int, 0x7FFF), -0x8000)

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        return raw_speed / steps_per_deg

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float | None = None,
        base_radius: float | None = None,
        max_raw: int | None = None,
    ) -> dict[str, int]:
        wheel_radius = wheel_radius or self.config.wheel_radius_m
        base_radius = base_radius or self.config.base_radius_m
        max_raw = max_raw or self.config.max_wheel_raw

        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])

        angles = np.radians(np.asarray(self.config.wheel_axis_angles_deg, dtype=float))
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]
        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed: int,
        back_wheel_speed: int,
        right_wheel_speed: int,
        wheel_radius: float | None = None,
        base_radius: float | None = None,
    ) -> dict[str, Any]:
        wheel_radius = wheel_radius or self.config.wheel_radius_m
        base_radius = base_radius or self.config.base_radius_m

        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * wheel_radius

        angles = np.radians(np.asarray(self.config.wheel_axis_angles_deg, dtype=float))
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)
        return {"x.vel": x, "y.vel": y, "theta.vel": theta}

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        start = time.perf_counter()
        base_wheel_vel = self.bus.sync_read("Present_Velocity", self.base_motors)
        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )
        obs_dict.update(base_vel)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read base velocities: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        base_goal_vel = {
            "x.vel": float(action.get("x.vel", 0.0)),
            "y.vel": float(action.get("y.vel", 0.0)),
            "theta.vel": float(action.get("theta.vel", 0.0)),
        }

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"],
            base_goal_vel["y.vel"],
            base_goal_vel["theta.vel"],
        )

        self.bus.sync_write("Goal_Velocity", base_wheel_goal_vel)
        return base_goal_vel

    def stop_base(self) -> None:
        self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        logger.info(f"{self} disconnected.")
