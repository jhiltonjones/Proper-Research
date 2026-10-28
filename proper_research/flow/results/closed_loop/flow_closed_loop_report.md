# Flow-fed closed-loop tracking (Layer 5 + DARE)

## Verdict

- **PASS** — Flow feedback vs perfect vision. flow-fed tip rms 0.172 mm vs vision 0.141 mm (gap +0.031 mm) -- the cost of replacing the camera with the ring.
- **PASS** — DARE + flow vs open loop. open loop (feedforward only, same start) never removes the initial offset -- final tip error 1.966 mm; the flow-fed DARE brings the final error to 0.033 mm.

Beam-tip tracking of the pre-computed L1->L2->L3 reference for the single-30-degree-bend lumen, with the tip error estimated from the **simulated four-element flow ring** rather than vision. The plant keeps contact enabled; the controller Jacobian is contact-free, so the flow-fed integral term has a genuine model mismatch to reject.

Flow feedback: a_ring = 0.90 mm, sigma_s/v_mean = 2.0% (assumption), 72 bpm.  DARE weights: tip 300, integral 2, tip-velocity 8, nullspace gain 0.1.


## Tracking results

| run | tip rms [mm] | tip max [mm] | tip final [mm] | tangent max [deg] | flow est. rms [mm] | held % |
|---|---|---|---|---|---|---|
| riccati_flow | 0.1722 | 0.6664 | 0.0334 | 24.22 | 0.0598 | 0.0 |
| riccati_vision | 0.1411 | 0.6664 | 0.0113 | 24.01 | nan | 0.0 |
| feedforward_only | 1.7809 | 2.0036 | 1.9662 | 29.97 | nan | 0.0 |

## Note on the resolved-rate baseline (`no_dare_flow`)

The plain `InverseJacobianBeamController` (resolved-rate, no DARE) is **not a
usable baseline with flow feedback** and diverges (tip error > 10 mm). Its
feedforward is *nullspace-projected* — it assumes the damped-least-squares task
term already carries the reference tip motion. A transverse-only flow
measurement makes the task term see only the (small) centring error, so the
along-track feedforward is projected out and the catheter fails to advance along
the trajectory.

The DARE controller is built around exactly this: it adds the **full** joint
feedforward `u_ref` explicitly and **locks the insertion axis** out of the
feedback, so the ring only ever corrects the transverse offset while the advance
follows the schedule. This is what the DARE structure buys here — not just an
optimal gain, but a control law whose feedforward survives a partial
measurement.

## Reading the numbers

* **Open loop fails.** With the 0.67 mm initial joint offset, feedforward alone
  never removes it and the tip drifts to ~2 mm — outside the 1.5 mm tolerance.
* **Flow feedback rescues it.** The DARE + simulated ring corrects the offset
  and tracks to 0.17 mm rms / 0.033 mm final, comfortably inside tolerance.
* **The cost of losing the camera is ~0.03 mm rms.** Perfect vision gives
  0.14 mm rms / 0.011 mm final; the flow ring's ~0.06 mm estimate noise adds
  little on top.
* **max = 0.666 mm for both DARE runs** is the initial offset itself, before the
  loop has corrected it — not a tracking excursion.
