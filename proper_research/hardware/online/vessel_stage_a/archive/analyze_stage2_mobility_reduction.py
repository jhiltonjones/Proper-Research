"""Stage-2 final figure: singular-value reduction under contact.

For J_cam, J_C, J_NC (all restricted to the x,y,rz source-pose columns
and XY tip rows, matching every Stage-1/2 comparison this session),
report sigma1/sigma2 and the mobility ratios

    R_contact = ||J_cam^contact||_F / ||J_cam^free||_F   (hardware)
    R_C       = ||J_C||_F / ||J_NC||_F                   (model, matched state)

and, the headline number, the per-mode singular-value reduction

    sigma_i^contact / sigma_i^free

for camera and model side by side.

Noise-justified significance threshold for a singular direction: this
session's 60 logged return-to-nominal repeatability checks give a
drift distribution with median 0.095mm, mean 0.19mm, p90 0.46mm
(calibration_2026-10-05/jcam_{source,contact}_*.json). Taking
sigma_tip ~= 0.2mm as a representative positioning-noise figure and
propagating through a finite-difference column (two independent
measurements, step size ~15-20mm for x/y or ~0.15-0.18rad for rz):

    sigma_col_noise ~= sqrt(2) * sigma_tip / eps
                     ~= 0.017 for x/y-type columns, ~0.002 for rz.

We use 0.015 as a round, slightly conservative floor on sigma_i, and
call a singular value "significant" if it clears
max(0.1*sigma_1, 0.015) -- the relative criterion from the user's
request, floored by the absolute noise estimate so a tiny sigma_1
doesn't let an equally-tiny sigma_2 count as a real second mode. Only
significant modes' directions are compared between matrices; a
non-significant sigma_2 is reported but excluded from any directional
claim.
"""
import numpy as np

from proper_research.hardware.online.vessel_stage_a.archive.analyze_jcam_contact_vs_models import (
    model_J_sources, camera_J,
)

SIGMA_NOISE_FLOOR = 0.015


def svd2(J):
    U, S, _ = np.linalg.svd(J)
    return U, S


def significant_mask(S):
    thresh = max(0.1 * S[0], SIGMA_NOISE_FLOOR)
    return S > thresh, thresh


def dir_angle(u, v):
    cos_t = np.clip(abs(np.dot(u, v)), -1, 1)
    return np.degrees(np.arccos(cos_t))


def main():
    for L_mm in (40.0, 50.0):
        import proper_research.hardware.online.vessel_stage_a.archive.analyze_jcam_contact_vs_models as m
        m.L_MM = L_mm
        for psi in (30.0, -30.0):
            print(f"\n{'='*76}\nL={L_mm:.0f}mm  psi={psi:+.0f}deg  (phi=35deg, r=225mm)\n{'='*76}")

            models = model_J_sources(psi)
            J_C = models["J_C^fast_source"][:2][:, [0, 1, 5]]
            J_NC = models["J_NC_source"][:2][:, [0, 1, 5]]
            J_cam_contact, _ = camera_J(psi, "contact")
            J_cam_free, _ = camera_J(psi, "source")

            mats = {
                "J_cam^free": J_cam_free, "J_cam^contact": J_cam_contact,
                "J_NC": J_NC, "J_C": J_C,
            }
            svds = {}
            print(f"{'matrix':>14s} {'sigma1':>9s} {'sigma2':>9s} {'sig1?':>6s} {'sig2?':>6s} {'||.||_F':>9s}  dominant_dir")
            for name, J in mats.items():
                U, S = svd2(J)
                sig_mask, thresh = significant_mask(S)
                fro = np.linalg.norm(J, "fro")
                svds[name] = (U, S, sig_mask)
                print(f"{name:>14s} {S[0]:9.5f} {S[1]:9.5f} {str(sig_mask[0]):>6s} {str(sig_mask[1]):>6s} {fro:9.5f}  {np.round(U[:,0],4)}")
            print(f"(significance threshold used: {thresh:.4f} = max(0.1*sigma1, {SIGMA_NOISE_FLOOR}))")

            print("\n--- mobility ratios ---")
            R_contact = np.linalg.norm(J_cam_contact, "fro") / np.linalg.norm(J_cam_free, "fro")
            R_C = np.linalg.norm(J_C, "fro") / np.linalg.norm(J_NC, "fro")
            print(f"R_contact = ||J_cam^contact||_F / ||J_cam^free||_F = {R_contact:.4f}")
            print(f"R_C       = ||J_C||_F / ||J_NC||_F (matched state)  = {R_C:.4f}")

            print("\n--- headline: per-mode singular-value reduction, camera vs model ---")
            U_free, S_free, mask_free = svds["J_cam^free"]
            U_contact, S_contact, mask_contact = svds["J_cam^contact"]
            U_NC, S_NC, mask_NC = svds["J_NC"]
            U_C, S_C, mask_C = svds["J_C"]
            print(f"{'mode':>6s} {'sigma_cam_c/sigma_cam_f':>24s} {'sigma_C/sigma_NC':>18s}")
            for i in range(2):
                r_cam = S_contact[i] / S_free[i] if S_free[i] > 1e-9 else float("nan")
                r_model = S_C[i] / S_NC[i] if S_NC[i] > 1e-9 else float("nan")
                tag = "" if (mask_free[i] and mask_contact[i]) else "  (sigma2 not significant on one/both sides -- interpret with caution)"
                print(f"{'σ'+str(i+1):>6s} {r_cam:24.4f} {r_model:18.4f}{tag}")

            print("\n--- directions, restricted to the significant-mode subspace ---")
            for name_a, name_b in (("J_cam^contact", "J_C"), ("J_cam^contact", "J_NC"), ("J_cam^free", "J_NC")):
                U_a, S_a, mask_a = svds[name_a]
                U_b, S_b, mask_b = svds[name_b]
                theta_dom = dir_angle(U_a[:, 0], U_b[:, 0])
                line = f"{name_a} vs {name_b}: dominant-dir angle = {theta_dom:.2f}deg"
                if mask_a[1] and mask_b[1]:
                    theta_weak = dir_angle(U_a[:, 1], U_b[:, 1])
                    line += f"  weak-dir angle = {theta_weak:.2f}deg (both sides have a significant 2nd mode)"
                else:
                    line += "  weak-dir comparison skipped (not significant on at least one side)"
                print(line)


if __name__ == "__main__":
    main()
