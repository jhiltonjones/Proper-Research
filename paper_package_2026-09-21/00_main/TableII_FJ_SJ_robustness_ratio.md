# Table II — Frozen-to-scheduled RMS ratio: INV-7 vs MPC

| Shape | INV-7 FJ/SJ RMS ratio | MPC FJ/SJ RMS ratio |
|---|---:|---:|
| Rectangle | 2.65 | 1.31 |
| Triangle (25 mm) | 1.20 | 1.04 |
| U-shape | 3.49 | NA — both attempted FJ trials safety-terminated |

Ratio = mean FJ RMS / mean SJ RMS for the same controller (a ratio of
condition means, not a run-paired ratio); >1 means the frozen Jacobian
degrades tracking relative to the scheduled Jacobian. MPC is more robust to
a frozen Jacobian on the rectangle and triangle (ratio closer to 1).
