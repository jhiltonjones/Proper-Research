from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np


JacobianVariant = Literal["contact", "no_contact"]
SolverMode = Literal["lti", "ltv_oneshot", "sqp_full"]


@dataclass(frozen=True)
class LumenConfig:
    bend_angle_deg: float
    length: float = 0.045
    n_pts: int = 130
    bend_start: float = 0.015
    bend_end: float = 0.025
    ds_target: float = 1e-3
    radius: float = 0.004

    def validate(self) -> None:
        if self.length <= 0.0:
            raise ValueError("LumenConfig.length must be positive.")

        if self.n_pts < 2:
            raise ValueError("LumenConfig.n_pts must be at least 2.")

        if self.ds_target <= 0.0:
            raise ValueError("LumenConfig.ds_target must be positive.")

        if self.radius <= 0.0:
            raise ValueError("LumenConfig.radius must be positive.")

        if not (0.0 <= self.bend_start < self.bend_end <= self.length):
            raise ValueError(
                "Require 0 <= bend_start < bend_end <= length. "
                f"Got bend_start={self.bend_start}, "
                f"bend_end={self.bend_end}, length={self.length}."
            )

    @property
    def bend_angle_rad(self) -> float:
        return float(np.deg2rad(self.bend_angle_deg))


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
    solver_mode: SolverMode = "sqp_full"
    rollout_steps: int = 1
    Np: int = 12
    N_sqp: int = 5
    max_steps: int = 200
    adaptive_rollout_enabled: bool = False

    def validate(self) -> None:
        allowed = {"lti", "ltv_oneshot", "sqp_full"}

        if self.solver_mode not in allowed:
            raise ValueError(
                f"solver_mode must be one of {sorted(allowed)}, "
                f"got {self.solver_mode!r}."
            )

        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be >= 1.")

        if self.Np < 1:
            raise ValueError("Np must be >= 1.")

        if self.rollout_steps > self.Np:
            raise ValueError(
                f"rollout_steps={self.rollout_steps} cannot exceed Np={self.Np}."
            )

        if self.N_sqp < 1:
            raise ValueError("N_sqp must be >= 1.")

        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")


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
        bend_token = _format_signed_int_token(self.lumen.bend_angle_deg)

        parts = [
            f"bend_{bend_token}",
            f"jac_{self.model.jacobian_variant}",
            f"plant_contact_{int(self.model.plant_contact)}",
            self.controller.solver_mode,
            f"rollout{self.controller.rollout_steps}",
            f"Np{self.controller.Np}",
            f"sqp{self.controller.N_sqp}",
        ]

        if self.seed:
            parts.append(f"seed{self.seed}")

        if self.tag:
            parts.append(_sanitize_token(self.tag))

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
    bend_angles_deg=(-30.0, -60.0, -90.0),
    jacobian_variants=("contact", "no_contact"),
    solver_modes=("sqp_full",),
    rollout_steps_values=(1,),
    Np: int = 12,
    N_sqp: int = 5,
    max_steps: int = 200,
    plant_contact: bool = True,
    tag: str = "",
    adaptive_rollout_enabled: bool = False,
) -> list[ExperimentConfig]:
    experiments = []

    for bend_angle in bend_angles_deg:
        for jacobian_variant in jacobian_variants:
            for solver_mode in solver_modes:
                for rollout_steps in rollout_steps_values:
                    exp = ExperimentConfig(
                        lumen=LumenConfig(
                            bend_angle_deg=float(bend_angle),
                        ),
                        model=ModelConfig(
                            plant_contact=bool(plant_contact),
                            jacobian_variant=jacobian_variant,
                            
                        ),
                        controller=ControllerRunConfig(
                            solver_mode=solver_mode,
                            rollout_steps=int(rollout_steps),
                            Np=int(Np),
                            N_sqp=int(N_sqp),
                            max_steps=int(max_steps),
                            adaptive_rollout_enabled = adaptive_rollout_enabled,
                        ),
                        run_root=Path(run_root),
                        tag=tag,
                    )

                    exp.validate()
                    experiments.append(exp)

    return experiments