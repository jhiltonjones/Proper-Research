from __future__ import annotations

import importlib
from pathlib import Path


PACKAGE_NAME = __package__ or (
    f"proper_research.simulation.{Path(__file__).resolve().parent.name}"
)
_smoke = importlib.import_module(f"{PACKAGE_NAME}.run_solver_smoke_test")


def main() -> None:
    # These two functions are also used by the smoke tests and benchmark.
    model = _smoke.build_run_plot_model(contact_enabled=True)
    p7 = _smoke.make_run_plot_p7()

    print(f"Package under test: {PACKAGE_NAME}")
    print("Shared source p7:", p7)

    result = model.solve(p7, commit=True)

    print("Tip position:", result.tip)
    print("Solver information:", result.info)

    model.plot_solution(
        p7,
        result=result,
        show=True,
    )


if __name__ == "__main__":
    main()