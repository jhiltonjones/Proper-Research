import json
import numpy as np
import matplotlib.pyplot as plt

JSON_PATHS = {
    "run5": "results_sweep_forward_validation5/sweep_results.json",
    # "run7": "results_sweep_forward_validation7/sweep_results.json",
}

plt.figure(figsize=(9, 9))
plt.plot(0.0, 0.0, "ko", markersize=8, label="Base / pivot")

styles = {
    "run5": {"beam": "-",  "pred": "o", "meas": "s"},
    "run7": {"beam": "--", "pred": "^", "meas": "d"},
}

for run_name, json_path in JSON_PATHS.items():
    with open(json_path, "r") as f:
        data = json.load(f)

    data = sorted(data, key=lambda r: r["j_idx"])

    first_pred_beam = True
    first_pred_tip = True
    first_meas_tip = True

    for row in data:
        j_idx = row["j_idx"]

        pred_beam = np.asarray(row["pred_beam_local_mm"], dtype=float)
        pred_tip = 1e3 * np.asarray(row["pred_tip_base_local_m"], dtype=float)
        meas_tip = 1e3 * np.asarray(row["meas_tip_base_local_m"], dtype=float)

        plt.plot(
            pred_beam[:, 0],
            pred_beam[:, 1],
            styles[run_name]["beam"],
            linewidth=1.3,
            alpha=0.9,
            label=f"{run_name} predicted beam" if first_pred_beam else None,
        )

        plt.plot(
            pred_tip[0],
            pred_tip[1],
            styles[run_name]["pred"],
            markersize=5,
            label=f"{run_name} predicted tip" if first_pred_tip else None,
        )

        plt.plot(
            meas_tip[0],
            meas_tip[1],
            styles[run_name]["meas"],
            markersize=5,
            label=f"{run_name} measured tip" if first_meas_tip else None,
        )

        plt.plot(
            [0.0, meas_tip[0]],
            [0.0, meas_tip[1]],
            "r--",
            alpha=0.2,
        )

        first_pred_beam = False
        first_pred_tip = False
        first_meas_tip = False

plt.xlabel("Base-local x [mm]")
plt.ylabel("Base-local y [mm]")
plt.title("Predicted beam reconstructions and measured tips (runs 5 and 7)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()