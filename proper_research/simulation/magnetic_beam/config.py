from __future__ import annotations

from dataclasses import dataclass

from .contact import ContactParams


@dataclass(frozen=True)
class BeamModelConfig:
    N_nodes: int = 10
    maxiter: int = 40
    L0_init: float = 0.01
    dL_internal: float = 0.04
    L_tip_full: float = 0.04
    L_tip_min: float = 0.01
    energy_scale: float = 1e-8
    u_scale: float = 30.0

    def validate(self) -> None:
        if self.N_nodes < 2:
            raise ValueError(f"N_nodes must be >= 2, got {self.N_nodes}.")
        if self.maxiter <= 0:
            raise ValueError(f"maxiter must be positive, got {self.maxiter}.")
        if self.L0_init <= 0:
            raise ValueError(f"L0_init must be positive, got {self.L0_init}.")
        if self.dL_internal <= 0:
            raise ValueError(f"dL_internal must be positive, got {self.dL_internal}.")
        if self.L_tip_full <= 0:
            raise ValueError(f"L_tip_full must be positive, got {self.L_tip_full}.")
        if self.L_tip_min <= 0:
            raise ValueError(f"L_tip_min must be positive, got {self.L_tip_min}.")
        if self.energy_scale <= 0:
            raise ValueError(f"energy_scale must be positive, got {self.energy_scale}.")
        if self.u_scale <= 0:
            raise ValueError(f"u_scale must be positive, got {self.u_scale}.")
        if self.L_tip_min > self.L_tip_full:
            raise ValueError(
                "L_tip_min should usually be <= L_tip_full. "
                f"Got L_tip_min={self.L_tip_min}, L_tip_full={self.L_tip_full}."
            )


@dataclass(frozen=True)
class ContactConfig:
    enabled: bool = True
    use_in_jacobian: bool = True
    params: ContactParams | None = None

    def validate(self) -> None:
        if self.enabled and self.params is None:
            raise ValueError("ContactConfig.params is required when enabled=True.")

        if self.params is not None:
            self.params.validate()

        if self.use_in_jacobian and not self.enabled:
            raise ValueError(
                "use_in_jacobian=True is inconsistent with enabled=False. "
                "Enable contact in the forward model or disable contact in the Jacobian."
            )

    @classmethod
    def enabled_default(cls, *, use_in_jacobian: bool = True) -> "ContactConfig":
        return cls(
            enabled=True,
            use_in_jacobian=use_in_jacobian,
            params=ContactParams(),
        )

    @classmethod
    def disabled(cls) -> "ContactConfig":
        return cls(
            enabled=False,
            use_in_jacobian=False,
            params=None,
        )