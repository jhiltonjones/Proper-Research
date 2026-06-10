from __future__ import annotations

import numpy as np

from proper_research.simulation.magnetic_beam.config import (
    BeamModelConfig,
    ContactConfig,
)
from proper_research.simulation.magnetic_beam.contact import (
    ContactParams,
    LumenQuery,
)
from proper_research.simulation.magnetic_beam.problem import BaseFrameConfig
from proper_research.simulation.magnetic_beam.forward_model import MagneticBeamForwardModel
from proper_research.simulation.magnetic_beam.diagnostics import summarise_solve_result


def make_test_lumen() -> LumenQuery:
    """
    Build a simple straight lumen along negative x.

    Shape convention:
        C: (M, 3)
        R: (M,)
    """
    x = np.linspace(0.0, -0.12, 80)
    C = np.column_stack(
        [
            x,
            np.zeros_like(x),
            np.zeros_like(x),
        ]
    )

    R = np.full(x.size, 0.010)

    return LumenQuery(C, R)


def make_test_Kinv_fun():
    """
    Return a simple constant compliance profile.

    This matches the old precompute_K_segments calling convention:

        Kinv_fun(smid, wire_len)

    Expected return shape:
        (3, 3, N_seg)

    where N_seg = len(smid).
    """
    Kinv_single = np.diag([1.0e-3, 1.0e-2, 1.0e-2])

    def Kinv_fun(smid, wire_len):
        smid = np.asarray(smid, float).reshape(-1)
        N_seg = smid.size

        Kinv = np.repeat(Kinv_single[:, :, None], N_seg, axis=2)

        return Kinv

    return Kinv_fun


def build_test_model(*, contact_enabled: bool = True) -> MagneticBeamForwardModel:
    base = BaseFrameConfig(
        p0_ur=np.array([0.0, 0.0, 0.0]),
        q0_ur=np.array([1.0, 0.0, 0.0, 0.0]),
        u_star=np.zeros(3),
    )

    beam = BeamModelConfig(
        N_nodes=10,
        maxiter=40,
        L0_init=0.01,
        dL_internal=0.02,
        L_tip_full=0.04,
        L_tip_min=0.01,
        energy_scale=1e-8,
        u_scale=30.0,
    )

    if contact_enabled:
        contact = ContactConfig(
            enabled=True,
            use_in_jacobian=True,
            params=ContactParams(
                r_beam=0.001,
                k=1e8,
                pen_switch=5e-5,
                k_hard=1e10,
                smooth=True,
                smooth_eps=1e-5,
                window=3,
            ),
        )
        lumen_query = make_test_lumen()
    else:
        contact = ContactConfig.disabled()
        lumen_query = None

    Kinv_fun = make_test_Kinv_fun()

    # Body-frame source dipole. Adjust magnitude to match your physical scale.
    m_body = np.array([0.0, 0.0, 1.0])

    return MagneticBeamForwardModel(
        base=base,
        beam=beam,
        contact=contact,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_query=lumen_query,
    )


def run_forward_only() -> None:
    model = build_test_model(contact_enabled=False)

    p7 = np.array(
        [
            0.03,   # source x
            0.00,   # source y
            0.03,   # source z
            0.0,    # rx
            0.0,    # ry
            0.0,    # rz
            0.06,   # insertion length
        ],
        dtype=float,
    )

    result = model.solve(p7, commit=True)

    print("\n=== FORWARD SOLVE TEST ===")
    summarise_solve_result(result)

    assert result.tip.shape == (3,)
    assert result.u_flat_opt.size == 3 * (model.beam.N_nodes - 1)
    assert model.cache.u_flat_opt is not None
    assert model.cache.tip is not None

    print("Forward solve smoke test passed.")


def run_forward_with_contact() -> None:
    model = build_test_model(contact_enabled=True)

    p7 = np.array(
        [
            0.03,
            0.00,
            0.03,
            0.0,
            0.0,
            0.0,
            0.06,
        ],
        dtype=float,
    )

    result = model.solve(p7, commit=True)

    print("\n=== FORWARD SOLVE WITH CONTACT TEST ===")
    summarise_solve_result(result)

    assert result.tip.shape == (3,)
    assert "W_cf" in result.parts
    assert model.cache.u_flat_opt is not None

    print("Forward solve with contact smoke test passed.")


def run_jacobian_test() -> None:
    model = build_test_model(contact_enabled=False)

    p7 = np.array(
        [
            0.03,
            0.00,
            0.03,
            0.0,
            0.0,
            0.0,
            0.06,
        ],
        dtype=float,
    )

    tip = model(p7)
    J = model.jacobian_tip_pose7(
        p7,
        solve_if_needed=False,
        eps_theta=1e-6,
        eps_hess=1e-4,
        debug_jac=False,
    )

    print("\n=== JACOBIAN TEST ===")
    print("tip =", tip)
    print("J shape =", J.shape)
    print(J)

    assert J.shape == (3, 7)
    assert np.all(np.isfinite(J))

    print("Jacobian smoke test passed.")


def main() -> None:
    print("Running magnetic_beam smoke tests...")

    run_forward_only()
    run_forward_with_contact()

    # Run this after the forward-only/contact tests pass.
    # If integrate_pq_and_sens_from_u has not been wired yet, this is the first
    # test expected to fail.
    run_jacobian_test()

    print("\nAll magnetic_beam smoke tests passed.")


if __name__ == "__main__":
    main()