#!/usr/bin/env python3
"""Check public and local/private dependencies required by the project."""

from cosserat_clean.external import dependency_status


def main() -> int:
    status = dependency_status()
    missing = []
    print("Dependency status")
    print("-----------------")
    for package, version in status.items():
        print(f"{package:34s} {version}")
        if version == "MISSING":
            missing.append(package)
    if missing:
        print("\nMissing:", ", ".join(missing))
        return 1
    print("\nAll dependencies are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
