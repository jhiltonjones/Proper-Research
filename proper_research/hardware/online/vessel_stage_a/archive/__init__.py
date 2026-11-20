"""One-off diagnostic/investigation/plotting scripts for vessel_stage_a,
moved here (2026-10-10) to keep the package's top level down to the
scripts someone actually runs as part of the regular workflow (build a
plan, run open-loop/inverse-jacobian/MPC on hardware). Nothing in this
subpackage is part of that workflow -- each file here is a dated,
one-shot investigation of a specific question (see each file's own
docstring for which). Safe to read for historical context; not meant to
be re-run as-is without checking whether the question it answered is
still relevant.
"""
