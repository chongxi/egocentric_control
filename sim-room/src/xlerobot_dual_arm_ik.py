#  Introduce about this IK main features : 
#  DualArmIKSolver: Differential Inverse Kinematics for Dual-Arm Robots
#
#  This module provides a robust, safety-aware differential IK solver for dual-arm mobile robots.
#  Key features:
#    - Supports simultaneous control of two 5-DOF arms with independent targets.
#    - Adaptive Damped Least Squares (DLS) solver with manipulability-aware damping to avoid singularities and reduce joint rumble.
#    - Safety gating: Automatically disables IK if target errors exceed configurable thresholds, preventing erratic motion.
#    - Optional low-pass filtering and joint step limiting for smooth, safe joint trajectories.
#    - Modular: Easily configurable for different robot kinematics, end-effectors, and control modes (position or pose).
#    - Efficient: Batched tensor operations for multi-environment simulation.
#    - Detailed logging and timing for performance monitoring and debugging.
#    - Designed for seamless integration with Isaac Sim and custom robot assets.




from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence, Tuple, TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaacsim.core.prims import XFormPrim

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation


RIGHT_ARM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
RIGHT_ARM_END_EFFECTOR_BODY = "Fixed_Jaw"
LEFT_ARM_JOINT_NAMES = ["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"]
LEFT_ARM_END_EFFECTOR_BODY = "Fixed_Jaw_2"
DAMPING_PARAM = 0.1
MAX_POSITION_ERROR = 0.5
MAX_ORIENTATION_ERROR = 1.0

@dataclass
class ArmIKDefinition:
    joint_names: Sequence[str]
    end_effector_body: str
    enable_orientation: bool = True
    ik_method: str = "dls"
    ik_params: dict[str, float] = field(default_factory=lambda: {"lambda_val": DAMPING_PARAM})

    @property
    def command_type(self) -> str:
        return "pose" if self.enable_orientation else "position"


class DualArmIKSolver:
    """Differential IK helper that drives both arms towards target markers."""

    def __init__(
        self,
        robot: "Articulation",
        left_target: XFormPrim | None,
        right_target: XFormPrim | None,
        device: str | torch.device | None = None,
        left_arm: ArmIKDefinition | None = None,
        right_arm: ArmIKDefinition | None = None,
        verbose: bool = True,
        max_position_error: float = MAX_POSITION_ERROR,  # Maximum position error in meters before IK is disabled
        max_orientation_error: float = MAX_ORIENTATION_ERROR,  # Maximum orientation error in radians
    ) -> None:
        self.robot = robot
        self.device = torch.device(device) if device is not None else robot.data.joint_pos.device
        self.num_envs = robot.num_instances
        self.left_target_prim = left_target
        self.right_target_prim = right_target
        self.verbose = verbose
        self.max_position_error = max_position_error
        self.max_orientation_error = max_orientation_error
        self.left_arm = left_arm or ArmIKDefinition(
            joint_names=LEFT_ARM_JOINT_NAMES,
            end_effector_body=LEFT_ARM_END_EFFECTOR_BODY,
        )
        self.right_arm = right_arm or ArmIKDefinition(
            joint_names=RIGHT_ARM_JOINT_NAMES,
            end_effector_body=RIGHT_ARM_END_EFFECTOR_BODY,
        )

        self._left_joint_ids, self._left_body_index = self._resolve_arm_handles(self.left_arm)
        self._right_joint_ids, self._right_body_index = self._resolve_arm_handles(self.right_arm)

        left_cfg = DifferentialIKControllerCfg(
            command_type=self.left_arm.command_type,
            use_relative_mode=False,
            ik_method=self.left_arm.ik_method,
            ik_params=self.left_arm.ik_params,
        )
        right_cfg = DifferentialIKControllerCfg(
            command_type=self.right_arm.command_type,
            use_relative_mode=False,
            ik_method=self.right_arm.ik_method,
            ik_params=self.right_arm.ik_params,
        )
        self._left_controller = DifferentialIKController(left_cfg, num_envs=self.num_envs, device=self.device)
        self._right_controller = DifferentialIKController(right_cfg, num_envs=self.num_envs, device=self.device)

        self._skip_next_step = True
        self.robot.update(0.0)
        self.sync_targets_to_current_pose()
        self._frame_count = 0
        self._ik_time_accumulator = 0.0
        self._ik_frame_count = 0

        # variables for low pass filter
        self._left_q_filt  = None
        self._right_q_filt = None

        # Safety gating statistics
        self._left_gate_count = 0
        self._right_gate_count = 0

    @property
    def left_joint_ids(self) -> torch.Tensor:
        return self._left_joint_ids

    @property
    def right_joint_ids(self) -> torch.Tensor:
        return self._right_joint_ids

    def reset(self) -> None:
        self.robot.update(0.0)
        self._left_controller.reset()
        self._right_controller.reset()
        self.sync_targets_to_current_pose()
        self._skip_next_step = True
        self._frame_count = 0
        self._ik_time_accumulator = 0.0
        self._ik_frame_count = 0

    def sync_targets_to_current_pose(self) -> None:
        if self.left_target_prim is not None:
            left_pose = self.robot.data.body_pose_w[:, self._left_body_index]
            self.left_target_prim.set_world_poses(left_pose[:, :3], left_pose[:, 3:7])
        if self.right_target_prim is not None:
            right_pose = self.robot.data.body_pose_w[:, self._right_body_index]
            self.right_target_prim.set_world_poses(right_pose[:, :3], right_pose[:, 3:7])

    def step(self) -> None:
        base_pose = self.robot.data.root_pose_w
        self._frame_count += 1

        # --- LEFT ARM: Get target and current end-effector pose in base frame ---
        left_target_pos_w, left_target_quat_w = self._get_target_world_pose(self.left_target_prim)
        left_target_pos_b, left_target_quat_b = subtract_frame_transforms(
            base_pose[:, 0:3], base_pose[:, 3:7], left_target_pos_w, left_target_quat_w
        )
        left_ee_pose_w = self.robot.data.body_pose_w[:, self._left_body_index]
        left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
            base_pose[:, 0:3], base_pose[:, 3:7], left_ee_pose_w[:, 0:3], left_ee_pose_w[:, 3:7]
        )
        left_command = self._compose_command(self.left_arm, left_target_pos_b, left_target_quat_b)
        self._left_controller.set_command(
            left_command,
            ee_pos=left_ee_pos_b if self.left_arm.command_type == "position" else None,
            ee_quat=left_ee_quat_b if self.left_arm.command_type == "position" else None,
        )

        # --- RIGHT ARM: Get target and current end-effector pose in base frame ---
        right_target_pos_w, right_target_quat_w = self._get_target_world_pose(self.right_target_prim)
        right_target_pos_b, right_target_quat_b = subtract_frame_transforms(
            base_pose[:, 0:3], base_pose[:, 3:7], right_target_pos_w, right_target_quat_w
        )
        right_ee_pose_w = self.robot.data.body_pose_w[:, self._right_body_index]
        right_ee_pos_b, right_ee_quat_b = subtract_frame_transforms(
            base_pose[:, 0:3], base_pose[:, 3:7], right_ee_pose_w[:, 0:3], right_ee_pose_w[:, 3:7]
        )
        right_command = self._compose_command(self.right_arm, right_target_pos_b, right_target_quat_b)
        self._right_controller.set_command(
            right_command,
            ee_pos=right_ee_pos_b if self.right_arm.command_type == "position" else None,
            ee_quat=right_ee_quat_b if self.right_arm.command_type == "position" else None,
        )

        # Skip the first step after reset to avoid instability
        if self._skip_next_step:
            self._skip_next_step = False
            return

        # Safety check: gate IK if errors are too large
        left_pos_error = torch.linalg.norm(left_target_pos_b - left_ee_pos_b, dim=1)
        right_pos_error = torch.linalg.norm(right_target_pos_b - right_ee_pos_b, dim=1)
        left_ori_error = self._quat_angle_error(left_target_quat_b, left_ee_quat_b)
        right_ori_error = self._quat_angle_error(right_target_quat_b, right_ee_quat_b)

        left_gate_active = (left_pos_error > self.max_position_error) | (left_ori_error > self.max_orientation_error)
        right_gate_active = (right_pos_error > self.max_position_error) | (right_ori_error > self.max_orientation_error)

        # Track gating events
        if left_gate_active.any():
            self._left_gate_count += 1
            if self.verbose and self._left_gate_count % 30 == 1:
                print(f"⚠️  [Left Arm Safety Gate] Target too far! pos_err={left_pos_error.mean().item():.3f}m, "
                      f"ori_err={left_ori_error.mean().item():.3f}rad (max: {self.max_position_error}m, {self.max_orientation_error:.2f}rad)")

        if right_gate_active.any():
            self._right_gate_count += 1
            if self.verbose and self._right_gate_count % 30 == 1:
                print(f"⚠️  [Right Arm Safety Gate] Target too far! pos_err={right_pos_error.mean().item():.3f}m, "
                      f"ori_err={right_ori_error.mean().item():.3f}rad (max: {self.max_position_error}m, {self.max_orientation_error:.2f}rad)")

        # If both arms are gated, skip IK entirely
        if left_gate_active.all() and right_gate_active.all():
            return

        # Start timing IK computation
        ik_start_time = time.perf_counter()

        # --- Compute Jacobians and solve IK for both arms ---
        jacobians = self.robot.root_physx_view.get_jacobians()
        left_ee_jacobi_idx = self._left_body_index - 1 if self.robot.is_fixed_base else self._left_body_index
        right_ee_jacobi_idx = self._right_body_index - 1 if self.robot.is_fixed_base else self._right_body_index

        # Left arm Jacobian and IK (only if not gated)
        if not left_gate_active.all():
            # float base  physx dofs is longer than the robot dofs so we need to offset the joint indices
            total_dofs = jacobians.shape[-1]
            base_dof_offset = max(total_dofs - len(self.robot.data.joint_names), 0)
            left_dof_indices = self._left_joint_ids + base_dof_offset if base_dof_offset else self._left_joint_ids
            left_jacobian = jacobians[:, left_ee_jacobi_idx, :, left_dof_indices].clone()
            left_joint_pos = self.robot.data.joint_pos[:, self._left_joint_ids]

            # apply the adaptive lambda to the controller
            left_adaptived_lambda = self._adaptive_lambda(left_jacobian[0], left_pos_error).mean().item()
            self._left_controller.cfg.ik_params["lambda_val"] = left_adaptived_lambda

            left_target = self._left_controller.compute(left_ee_pos_b, left_ee_quat_b, left_jacobian, left_joint_pos)
            left_target = torch.where(torch.isfinite(left_target), left_target, left_joint_pos)
            self.robot.set_joint_position_target(left_target, joint_ids=self._left_joint_ids)
        else:
            # Keep current position when gated
            left_adaptived_lambda = 0.0
            left_jacobian = None

        # Right arm Jacobian and IK (only if not gated)
        if not right_gate_active.all():
            total_dofs = jacobians.shape[-1]
            base_dof_offset = max(total_dofs - len(self.robot.data.joint_names), 0)
            right_dof_indices = self._right_joint_ids + base_dof_offset if base_dof_offset else self._right_joint_ids
            right_jacobian = jacobians[:, right_ee_jacobi_idx, :, right_dof_indices].clone()
            right_joint_pos = self.robot.data.joint_pos[:, self._right_joint_ids]

            # apply the adaptive lambda to the controller for right arm
            right_adaptived_lambda = self._adaptive_lambda(right_jacobian[0], right_pos_error).mean().item()
            self._right_controller.cfg.ik_params["lambda_val"] = right_adaptived_lambda

            right_target = self._right_controller.compute(right_ee_pos_b, right_ee_quat_b, right_jacobian, right_joint_pos)
            right_target = torch.where(torch.isfinite(right_target), right_target, right_joint_pos)
            self.robot.set_joint_position_target(right_target, joint_ids=self._right_joint_ids)
        else:
            # Keep current position when gated
            right_adaptived_lambda = 0.0
            right_jacobian = None

        # End timing and accumulate
        ik_end_time = time.perf_counter()
        ik_duration = ik_end_time - ik_start_time
        self._ik_time_accumulator += ik_duration
        self._ik_frame_count += 1

        # Logging
        if left_jacobian is not None and right_jacobian is not None:
            self._log(
                jacobians, left_ee_jacobi_idx, right_ee_jacobi_idx,
                left_jacobian, right_jacobian,
                left_target_pos_b, left_ee_pos_b, left_target_quat_b, left_ee_quat_b,
                right_target_pos_b, right_ee_pos_b, right_target_quat_b, right_ee_quat_b,
                left_adaptived_lambda, right_adaptived_lambda
            )


# low pass filter to smooth the joint position ,default is off
    def _smooth_and_limit(self, q_prev: torch.Tensor | None, q_cur: torch.Tensor, q_next: torch.Tensor, max_joint_step=0.02,alpha_q=0.8):
        if q_prev is None:
            q_filt = q_next
        else:
            q_filt = alpha_q * q_next + (1.0 - alpha_q) * q_prev
        dq = (q_filt - q_cur).clamp(min=-max_joint_step, max=max_joint_step)
        return q_cur + dq, q_filt

# apply the low pass filter to the target exmaple code here 
            # left_target_raw = self._left_controller.compute(left_ee_pos_b, left_ee_quat_b, left_jacobian, left_joint_pos)
            # left_target_raw = torch.where(torch.isfinite(left_target_raw), left_target_raw, left_joint_pos)
            # left_target_saturated, self._left_q_filt = self._smooth_and_limit(self._left_q_filt, left_joint_pos, left_target_raw)
            # self.robot.set_joint_position_target(left_target_saturated, joint_ids=self._left_joint_ids)


# adaptive lambda to avoid singularity and rumble, default is on 
    def _adaptive_lambda(self, J: torch.Tensor,pos_err_mag: torch.Tensor,*,
        base_lambda: float = 0.1,   # small, always-on damping
        lam_max: float = 0.8,       # cap (adjust after you see variation)
        k_manip: float = 0.10,       # manipulability sensitivity
        k_err: float = 0.20,         # error sensitivity
        sigma_ref: float = 0.20,     # "healthy" σ_min
        manip_slope: float = 0.08,   # softness of manipulability curve
        err_cap: float = 0.30,       # 50 cm → avoids instant saturation
        use_position_rows: bool = True,
        log_debug: bool = False
    ) -> torch.Tensor:
        """
        Adaptive DLS damping using a smooth manipulability term (via softplus) and a capped
        error term. Accepts J of shape [6,dof]/[B,6,dof] or [3,dof]/[B,3,dof].
        Returns scalar (unbatched) or [B] (batched).
        """
        # ---- normalize shapes ----
        batched = (J.dim() == 3)
        if not batched:
            J = J.unsqueeze(0)  # [1, r, dof]

        # ---- choose rows (prefer 3xDOF for 5-DOF arm) ----
        if use_position_rows and J.shape[1] >= 6:
            Jt = J[:, 0:3, :]
        else:
            Jt = J

        device, dtype = Jt.device, Jt.dtype

        # ---- singular values and σ_min ----
        svals = torch.linalg.svdvals(Jt)          # [B, min(r,dof)]
        sigma_min = svals.min(dim=1).values       # [B]

        # ---- constants as tensors (fixes the softplus float error) ----
        sigma_ref_t   = torch.as_tensor(sigma_ref,   device=device, dtype=dtype)
        manip_slope_t = torch.as_tensor(manip_slope, device=device, dtype=dtype)
        err_cap_t     = torch.as_tensor(err_cap,     device=device, dtype=dtype)
        base_lambda_t = torch.as_tensor(base_lambda, device=device, dtype=dtype)

        # ---- manipulability term (smooth, bounded, ~0 when σ>=sigma_ref) ----
        # x grows as sigma_min drops below sigma_ref; softplus keeps it gentle
        x = (sigma_ref_t - sigma_min) / manip_slope_t
        num   = torch.nn.functional.softplus(x)
        denom = torch.nn.functional.softplus(sigma_ref_t / manip_slope_t) + 1e-6
        manip_term = (num / denom).clamp(0.0, 1.0)    # [B]

        # ---- error term (0→1 over 0→err_cap, then capped) ----
        if pos_err_mag.dim() == 0:
            pos_err_mag = pos_err_mag.expand(Jt.shape[0])
        elif pos_err_mag.dim() > 1:
            pos_err_mag = pos_err_mag.reshape(-1)
        err_term = (pos_err_mag / err_cap_t).clamp(0.0, 1.0)  # [B]

        # ---- assemble λ and clamp ----
        lam = base_lambda_t + k_manip * manip_term + k_err * err_term
        lam = lam.clamp(max=lam_max)                           # [B]

        if log_debug:
            print(f"[AdaptiveLambdaDBG] "
                f"sigma_min_mean={sigma_min.mean().item():.4f} "
                f"manip_term_mean={manip_term.mean().item():.3f} "
                f"err_term_mean={err_term.mean().item():.3f} "
                f"lambda_mean={lam.mean().item():.3f}")

        return lam if batched else lam.squeeze(0)





    def _log(
        self,
        jacobians: torch.Tensor,
        left_ee_jacobi_idx: int,
        right_ee_jacobi_idx: int,
        left_jacobian: torch.Tensor,
        right_jacobian: torch.Tensor,
        left_target_pos_b: torch.Tensor,
        left_ee_pos_b: torch.Tensor,
        left_target_quat_b: torch.Tensor,
        left_ee_quat_b: torch.Tensor,
        right_target_pos_b: torch.Tensor,
        right_ee_pos_b: torch.Tensor,
        right_target_quat_b: torch.Tensor,
        right_ee_quat_b: torch.Tensor,
        left_adaptived_lambda: float,
        right_adaptived_lambda: float,
    ) -> None:
        """Log diagnostics for IK solver (first 10 frames or every 30 frames)."""
        # Determine if we should print this frame
        should_print = self.verbose and (self._frame_count <= 10 or self._frame_count % 30 == 0)
        if not should_print:
            return

        # Calculate average IK time for the window
        avg_ik_time_ms = (self._ik_time_accumulator / self._ik_frame_count * 1000.0) if self._ik_frame_count > 0 else 0.0

        print(f"\n{'='*80}")
        print(f"[IK] Frame {self._frame_count}")
        print(f"{'='*80}")
        print(f"[Performance] Avg IK time (last {self._ik_frame_count} frames): {avg_ik_time_ms:.4f}ms")
        print(f"[Adaptive Lambda] Left: {left_adaptived_lambda}, Right: {right_adaptived_lambda}")

        # Check articulation state
        print(f"[State] is_fixed_base={self.robot.is_fixed_base}, bodies={jacobians.shape[1]}, DOFs={jacobians.shape[3]}")

        # Check for base drift
        base_pos = self.robot.data.root_pos_w[0]
        base_quat = self.robot.data.root_pose_w[0, 3:7]
        print(f"[Base] pos=[{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}], "
              f"quat=[{base_quat[0]:.4f}, {base_quat[1]:.4f}, {base_quat[2]:.4f}, {base_quat[3]:.4f}]")

        # Check for NaN in Jacobian
        if torch.isnan(jacobians).any():
            nan_count = torch.isnan(jacobians).sum().item()
            print(f"⚠️  WARNING: {nan_count} NaN values in Jacobian!")

        # Left arm diagnostics
        print(f"\n[Left Arm] EE body idx={self._left_body_index}, jacobi_idx={left_ee_jacobi_idx}")
        print(f"[Left Arm] Joint IDs: {self._left_joint_ids.cpu().tolist()}")
        print(f"[Left Arm] Jacobian shape: {left_jacobian.shape}")

        # Check Jacobian condition for left arm
        with torch.no_grad():
            JJT = left_jacobian[0] @ left_jacobian[0].T
            try:
                det = torch.linalg.det(JJT).item()
                print(f"[Left Arm] J@J^T determinant: {det:.6e}")
                if abs(det) < 1e-6:
                    print(f"⚠️  WARNING: Near-singular matrix!")
            except Exception as e:
                print(f"⚠️  Could not compute determinant: {e}")

        # Right arm diagnostics
        print(f"\n[Right Arm] EE body idx={self._right_body_index}, jacobi_idx={right_ee_jacobi_idx}")
        print(f"[Right Arm] Joint IDs: {self._right_joint_ids.cpu().tolist()}")
        print(f"[Right Arm] Jacobian shape: {right_jacobian.shape}")

        # Tracking errors
        left_pos_err = torch.linalg.norm(left_target_pos_b - left_ee_pos_b, dim=1)
        right_pos_err = torch.linalg.norm(right_target_pos_b - right_ee_pos_b, dim=1)
        left_ori_err = self._quat_angle_error(left_target_quat_b, left_ee_quat_b)
        right_ori_err = self._quat_angle_error(right_target_quat_b, right_ee_quat_b)

        print(f"\n[Tracking Error]")
        print(f"  Left:  pos_err={left_pos_err.mean().item():.4f}m, ori_err={left_ori_err.mean().item():.3f}rad")
        print(f"  Right: pos_err={right_pos_err.mean().item():.4f}m, ori_err={right_ori_err.mean().item():.3f}rad")

        # Body name verification
        try:
            left_body_name = self.robot.data.body_names[self._left_body_index]
            right_body_name = self.robot.data.body_names[self._right_body_index]
            print(f"\n[Body Names] Left: '{left_body_name}', Right: '{right_body_name}'")
        except Exception:
            pass

        # Joint mapping verification
        if hasattr(self.robot.data, "joint_names"):
            print(f"\n[Joint Mapping]")
            left_id_list = self._left_joint_ids.detach().cpu().tolist()
            right_id_list = self._right_joint_ids.detach().cpu().tolist()

            print("  Left arm:")
            for i, jid in enumerate(left_id_list):
                resolved = self.robot.data.joint_names[jid]
                expected = self.left_arm.joint_names[i] if i < len(self.left_arm.joint_names) else "<n/a>"
                match = "✓" if resolved == expected else "✗"
                print(f"    [{i}] id={jid} resolved='{resolved}' expected='{expected}' {match}")

            print("  Right arm:")
            for i, jid in enumerate(right_id_list):
                resolved = self.robot.data.joint_names[jid]
                expected = self.right_arm.joint_names[i] if i < len(self.right_arm.joint_names) else "<n/a>"
                match = "✓" if resolved == expected else "✗"
                print(f"    [{i}] id={jid} resolved='{resolved}' expected='{expected}' {match}")

        print(f"{'='*80}\n")

        # Reset timing accumulator for next window
        self._ik_time_accumulator = 0.0
        self._ik_frame_count = 0


    def _compose_command(
        self,
        arm: ArmIKDefinition,
        target_pos_b: torch.Tensor,
        target_quat_b: torch.Tensor,
    ) -> torch.Tensor:
        if arm.command_type == "pose":
            return torch.cat((target_pos_b, target_quat_b), dim=1)
        return target_pos_b

    def _resolve_arm_handles(
        self,
        arm: ArmIKDefinition,
    ) -> Tuple[torch.Tensor, int]:
        joint_ids, resolved_names = self.robot.find_joints(arm.joint_names, preserve_order=True)
        if len(joint_ids) != len(arm.joint_names):
            missing = set(arm.joint_names) - set(resolved_names)
            raise RuntimeError(f"Failed to resolve joints {missing} for end-effector '{arm.end_effector_body}'.")
        joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=self.device)

        body_indices, _ = self.robot.find_bodies(arm.end_effector_body)
        if not body_indices:
            raise RuntimeError(f"Failed to resolve end-effector body '{arm.end_effector_body}'.")
        return joint_ids_tensor, body_indices[0]

    def _quat_angle_error(self, target_quat_b: torch.Tensor, current_quat_b: torch.Tensor) -> torch.Tensor:
        target = torch.nn.functional.normalize(target_quat_b, dim=1)
        current = torch.nn.functional.normalize(current_quat_b, dim=1)
        dot = (target * current).sum(dim=1).abs().clamp(min=0.0, max=1.0)
        return 2.0 * torch.arccos(dot)

    def _get_target_world_pose(self, prim: XFormPrim | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if prim is None:
            zeros_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            unit_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
            unit_quat[:, 0] = 1.0
            return zeros_pos, unit_quat
        pos, quat = prim.get_world_poses()
        if isinstance(pos, torch.Tensor):
            pos_t = pos.detach().clone().to(device=self.device, dtype=torch.float32)
        else:
            pos_t = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
        if isinstance(quat, torch.Tensor):
            quat_t = quat.detach().clone().to(device=self.device, dtype=torch.float32)
        else:
            quat_t = torch.as_tensor(quat, dtype=torch.float32, device=self.device)
        return pos_t, quat_t


__all__ = ["ArmIKDefinition", "DualArmIKSolver"]
