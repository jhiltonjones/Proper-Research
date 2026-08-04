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
    # L-BFGS-B settings.  The old 1e-22/1e-14 tolerances forced many
    # numerically meaningless iterations on an energy that is already scaled.
    optimizer_ftol: float = 1e-8
    optimizer_gtol: float = 1e-5
    optimizer_maxls: int = 80
    optimizer_maxfun: int = 50000
    optimizer_maxcor: int = 20
    # SciPy can report max-iteration/line-search failure even when the final
    # projected gradient is already adequate for controller use.  Accept only
    # a genuinely stationary result; never accept based on iteration count
    # alone.
    accept_stationary_failure: bool = True
    acceptable_grad_inf_scaled: float = 1e-3
    diagnose_failed_gradient: bool = True
    gradient_check_eps: float = 1e-6
    # A controller normally changes the actuation by a small amount between
    # frames.  In that case solve the requested final length directly from the
    # previous equilibrium; retain continuation for cold starts.
    direct_warm_start: bool = True
    fallback_to_continuation: bool = True
    exact_cache_atol: float = 1e-12

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
        if self.optimizer_ftol <= 0:
            raise ValueError("optimizer_ftol must be positive.")
        if self.optimizer_gtol <= 0:
            raise ValueError("optimizer_gtol must be positive.")
        if self.optimizer_maxls <= 0:
            raise ValueError("optimizer_maxls must be positive.")
        if self.optimizer_maxfun <= 0:
            raise ValueError("optimizer_maxfun must be positive.")
        if self.optimizer_maxcor <= 0:
            raise ValueError("optimizer_maxcor must be positive.")
        if self.acceptable_grad_inf_scaled <= 0:
            raise ValueError("acceptable_grad_inf_scaled must be positive.")
        if self.gradient_check_eps <= 0:
            raise ValueError("gradient_check_eps must be positive.")
        if self.exact_cache_atol < 0:
            raise ValueError("exact_cache_atol must be non-negative.")
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