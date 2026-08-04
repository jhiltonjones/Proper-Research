from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np



JacobianVariant = Literal["contact", "no_contact"]

ControllerKind = Literal[
    "mpc",
    "inverse_jacobian",
]

SolverMode = Literal[
    "lti",
    "ltv_oneshot",
    "sqp_full",
]

InverseSequenceMode = Literal[
    "held",
    "rollout_lti",
    "rollout_ltv",
]
JacobianVariant = Literal["contact", "no_contact"]
SolverMode = Literal["lti", "ltv_oneshot", "sqp_full"]


from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LumenBend:
    bend_axis: tuple[float, float, float]
    bend_angle_rad: float
    bend_start: float
    bend_end: float

    def validate(self, lumen_length: float) -> None:
        axis = np.asarray(self.bend_axis, float)

        if axis.shape != (3,):
            raise ValueError("bend_axis must contain three values.")

        if np.linalg.norm(axis) <= 1e-12:
            raise ValueError("bend_axis cannot be zero.")

        if not (
            0.0
            <= self.bend_start
            < self.bend_end
            <= lumen_length
        ):
            raise ValueError(
                "Require 0 <= bend_start < bend_end <= lumen length."
            )


@dataclass(frozen=True)
class LumenConfig:
    length: float = 0.045
    n_pts: int = 130
    n_ref_pts: int = 40
    ds_target: float = 1e-3
    radius: float = 0.004
    bends: tuple[LumenBend, ...] = ()

    def validate(self) -> None:
        if self.length <= 0.0:
            raise ValueError("Lumen length must be positive.")

        if self.n_pts < 2:
            raise ValueError("n_pts must be at least two.")

        if self.n_ref_pts < 2:
            raise ValueError("n_ref_pts must be at least two.")

        if self.ds_target <= 0.0:
            raise ValueError("ds_target must be positive.")

        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")

        for bend in self.bends:
            bend.validate(self.length)

    @property
    def bend_label(self) -> str:
        if not self.bends:
            return "straight"

        angles = [
            np.rad2deg(bend.bend_angle_rad)
            for bend in self.bends
        ]

        return "_".join(
            f"{angle:+.0f}"
            for angle in angles
        )
    @property
    def bend_angles_deg(self) -> tuple[float, ...]:
        return tuple(
            float(np.rad2deg(bend.bend_angle_rad))
            for bend in self.bends
        )
@dataclass(frozen=True)
class ModelConfig:
    plant_contact: bool = True
    jacobian_variant: JacobianVariant = "contact"

    def validate(self) -> None:
        allowed = {"contact", "no_contact"}

        if self.jacobian_variant not in allowed:
            raise ValueError(
                f"jacobian_variant must be one of {sorted(allowed)}, "
                f"got {self.jacobian_variant!r}."
            )


@dataclass(frozen=True)
class ControllerRunConfig:
    # Controller family
    controller_kind: ControllerKind = "mpc"

    # MPC-specific configuration
    solver_mode: SolverMode = "sqp_full"
    N_sqp: int = 5

    # Shared horizon/execution configuration
    rollout_steps: int = 1
    Np: int = 12
    max_steps: int = 200
    adaptive_rollout_enabled: bool = False

    # Inverse-Jacobian configuration
    inverse_sequence_mode: InverseSequenceMode = "rollout_ltv"

    # Fraction of the current/predicted tip error requested per stage.
    inverse_kp: float = 1

    # Maximum requested one-step tip displacement.
    inverse_desired_step_max_m: float = 4.0e-3

    # Additional regularisation added to the inverse-Jacobian solve.
    inverse_extra_damping: float = 0.0

    def validate(self) -> None:
        controller_kinds = {
            "mpc",
            "inverse_jacobian",
        }

        if self.controller_kind not in controller_kinds:
            raise ValueError(
                f"controller_kind must be one of "
                f"{sorted(controller_kinds)}, "
                f"got {self.controller_kind!r}."
            )

        if self.controller_kind == "mpc":
            solver_modes = {
                "lti",
                "ltv_oneshot",
                "sqp_full",
            }

            if self.solver_mode not in solver_modes:
                raise ValueError(
                    f"solver_mode must be one of "
                    f"{sorted(solver_modes)}, "
                    f"got {self.solver_mode!r}."
                )

        inverse_modes = {
            "held",
            "rollout_lti",
            "rollout_ltv",
        }

        if self.inverse_sequence_mode not in inverse_modes:
            raise ValueError(
                f"inverse_sequence_mode must be one of "
                f"{sorted(inverse_modes)}, "
                f"got {self.inverse_sequence_mode!r}."
            )

        if self.rollout_steps < 1:
            raise ValueError(
                "rollout_steps must be >= 1."
            )

        if self.Np < 1:
            raise ValueError("Np must be >= 1.")

        if self.rollout_steps > self.Np:
            raise ValueError(
                f"rollout_steps={self.rollout_steps} "
                f"cannot exceed Np={self.Np}."
            )

        if self.N_sqp < 1:
            raise ValueError("N_sqp must be >= 1.")

        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        if self.inverse_kp < 0.0:
            raise ValueError(
                "inverse_kp must be non-negative."
            )

        if (
            self.inverse_desired_step_max_m is not None
            and self.inverse_desired_step_max_m <= 0.0
        ):
            raise ValueError(
                "inverse_desired_step_max_m must be "
                "positive or None."
            )

        if self.inverse_extra_damping < 0.0:
            raise ValueError(
                "inverse_extra_damping must be non-negative."
            )

@dataclass(frozen=True)
class ExperimentConfig:
    lumen: LumenConfig
    model: ModelConfig
    controller: ControllerRunConfig
    run_root: Path
    tag: str = ""
    seed: int = 0

    def validate(self) -> None:
        self.lumen.validate()
        self.model.validate()
        self.controller.validate()

        if not isinstance(self.run_root, Path):
            raise TypeError("run_root must be a pathlib.Path.")

        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    @property
    def run_name(self) -> str:
        bend_angles = self.lumen.bend_angles_deg

        if bend_angles:
            bend_tokens = [
                _format_signed_int_token(angle)
                for angle in bend_angles
            ]

            bend_part = (
                "bends_" + "_".join(bend_tokens)
            )
        else:
            bend_part = "bends_straight"

        def _format_float_token(x: float) -> str:
            text = f"{float(x):g}"
            return (
                text
                .replace("-", "m")
                .replace(".", "p")
            )

        parts = [
            bend_part,
            f"jac_{self.model.jacobian_variant}",
            f"plant_contact_{int(self.model.plant_contact)}",
            f"ctrl_{self.controller.controller_kind}",
        ]

        if self.controller.controller_kind == "mpc":
            parts.extend(
                [
                    self.controller.solver_mode,
                    f"rollout{self.controller.rollout_steps}",
                    f"Np{self.controller.Np}",
                    f"sqp{self.controller.N_sqp}",
                ]
            )

        elif (
            self.controller.controller_kind
            == "inverse_jacobian"
        ):
            kp_token = _format_float_token(
                self.controller.inverse_kp
            )

            parts.extend(
                [
                    self.controller.inverse_sequence_mode,
                    f"kp{kp_token}",
                    f"rollout{self.controller.rollout_steps}",
                    f"Np{self.controller.Np}",
                ]
            )

        if self.seed:
            parts.append(f"seed{self.seed}")

        if self.tag:
            parts.append(
                _sanitize_token(self.tag)
            )

        return "_".join(parts)

    @property
    def out_root(self) -> Path:
        return self.run_root / self.run_name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_root"] = str(self.run_root)
        d["run_name"] = self.run_name
        d["out_root"] = str(self.out_root)
        return d


def _format_signed_int_token(x: float) -> str:
    """
    Convert signed angle to filesystem-safe token.

    Examples:
        -30.0 -> m30
        +30.0 -> p30
          0.0 -> p0
    """
    xi = int(round(float(x)))

    if xi < 0:
        return f"m{abs(xi)}"

    return f"p{xi}"


def _sanitize_token(s: str) -> str:
    """
    Basic filesystem-safe token.
    """
    s = str(s).strip()

    if not s:
        return ""

    keep = []

    for ch in s:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")

    return "".join(keep)
def make_curvature_jacobian_grid(
    *,
    run_root: Path,
    lumen_configs: tuple[LumenConfig, ...],
    jacobian_variants=("contact", "no_contact"),
    controller_kinds=("mpc",),
    solver_modes=("sqp_full",),
    inverse_sequence_modes=("rollout_ltv",),
    inverse_kp_values=(0.3,),
    inverse_desired_step_max_m=4.0e-3,
    inverse_extra_damping=0.0,
    rollout_steps_values=(1,),
    Np: int = 12,
    N_sqp: int = 5,
    max_steps: int = 200,
    plant_contact: bool = True,
    tag: str = "",
    adaptive_rollout_enabled: bool = False,
) -> list[ExperimentConfig]:
    experiments: list[ExperimentConfig] = []

    for lumen_cfg in lumen_configs:
        lumen_cfg.validate()

        for jacobian_variant in jacobian_variants:
            for controller_kind in controller_kinds:
                for rollout_steps in rollout_steps_values:

                    if controller_kind == "mpc":
                        controller_variants = [
                            {
                                "solver_mode": solver_mode,
                                "inverse_sequence_mode": "rollout_ltv",
                                "inverse_kp": 0.3,
                            }
                            for solver_mode in solver_modes
                        ]

                    elif controller_kind == "inverse_jacobian":
                        controller_variants = [
                            {
                                "solver_mode": "lti",
                                "inverse_sequence_mode": mode,
                                "inverse_kp": float(kp),
                            }
                            for mode in inverse_sequence_modes
                            for kp in inverse_kp_values
                        ]

                    else:
                        raise ValueError(
                            f"Unknown controller_kind: "
                            f"{controller_kind!r}."
                        )

                    for variant in controller_variants:
                        lumen_tag = (
                            f"bends_{lumen_cfg.bend_label}"
                        )

                        combined_tag = "_".join(
                            part
                            for part in (tag, lumen_tag)
                            if part
                        )

                        exp = ExperimentConfig(
                            lumen=lumen_cfg,
                            model=ModelConfig(
                                plant_contact=bool(
                                    plant_contact
                                ),
                                jacobian_variant=(
                                    jacobian_variant
                                ),
                            ),
                            controller=ControllerRunConfig(
                                controller_kind=(
                                    controller_kind
                                ),
                                solver_mode=variant[
                                    "solver_mode"
                                ],
                                rollout_steps=int(
                                    rollout_steps
                                ),
                                Np=int(Np),
                                N_sqp=int(N_sqp),
                                max_steps=int(max_steps),
                                adaptive_rollout_enabled=(
                                    adaptive_rollout_enabled
                                    if controller_kind == "mpc"
                                    else False
                                ),
                                inverse_sequence_mode=variant[
                                    "inverse_sequence_mode"
                                ],
                                inverse_kp=variant[
                                    "inverse_kp"
                                ],
                                inverse_desired_step_max_m=(
                                    inverse_desired_step_max_m
                                ),
                                inverse_extra_damping=(
                                    inverse_extra_damping
                                ),
                            ),
                            run_root=Path(run_root),
                            tag=combined_tag,
                        )

                        exp.validate()
                        experiments.append(exp)

    return experiments