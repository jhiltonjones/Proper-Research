"""Advancer-excitation identification: does smooth commanded insertion excite
real beam-tip vibration that the feedback controller later chases?

See this package's README.md for the full protocol and the pipeline:
``acquire.py`` (live, robot fixed) -> ``analyze.py`` (ring-down / PSD, no
robot) -> ``causality.py`` (offline INV replay + spectral comparison against
an existing closed-loop run, no robot).
"""
