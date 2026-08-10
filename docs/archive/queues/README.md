# Archived run queues and smoke artifacts

This directory preserves the provenance of the ad-hoc execution queues that
drove the StateKV retest, external-validity (extval), corner-gate, and
open-search experiment batches in August 2026, plus the smoke-run
configurations and outputs used to validate each configuration before the
full runs.  Everything here was recovered from the gitignored `tmp/` scratch
tree during the 2026-08-10 code audit (see `docs/CODE_AUDIT.md`) and is kept
read-only for lineage: the `*_queue*.sh` scripts and their `.log` outputs
record exactly which commands ran in which order, the `*_smoke_config.yaml`
files (with `r2b_smoke_trigger_rule.json`) record the pre-flight
configurations, and the `*_smoke_run/` directories hold the small smoke
outputs those configs produced.  Nothing here is an input to current
analysis; the canonical results live under `results/` and `analysis/`.
