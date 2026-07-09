# ADR-007: Evaluation Subsystem

## Status

Accepted.

## Context

D6 already produces trained weights under `models/trained/`. D7 needs to answer a
different question: given one trained weight and one dataset YAML, how well does
the model perform?

The evaluation flow looks similar to training, but it should remain smaller. It
can reuse the existing runtime config, reference resolvers, logging helpers, and
metric container instead of inventing new infrastructure.

## Decision

Add `od_platform.evaluation` as a thin orchestration layer:

- `ValService.evaluate()` builds D5 val config with `build_val_config()`.
- Trained weights are resolved with `resolve_trained_model()` and fail fast when
  missing.
- Dataset YAMLs are resolved with the shared dataset path helper.
- Metrics reuse `TrainMetrics` through the alias `ValMetrics`.
- Results write `odp_audit.json` with `kind: val`.
- CLI entry point is `odp-val`, separate from D4 `odp-validate`.

Evaluation does not create a new long-lived weight artifact. `ValResult` has no
`best_weight` field and the service only records metrics, logs, and an audit
snapshot.

## Consequences

Positive:

- D7 adds only `evaluation/service.py`, `evaluation/__init__.py`, and a small CLI.
- The module depends on common infrastructure, not on the training package.
- Missing private weights fail immediately instead of falling through to
  Ultralytics model lookup or downloads.

Tradeoffs:

- The metric type name is an alias, so display strings in `TrainMetrics` still
  come from the shared training/evaluation metric renderer.
- Tests must protect the service boundary because most of the behavior is
  orchestration around other modules.

## Acceptance

- `from od_platform.evaluation import ValService, ValMetrics` works.
- `ValMetrics is TrainMetrics` is true.
- `odp-val --help` exposes the model evaluation command.
- Unit tests cover success, missing weight fail-fast, exception handling,
  audit `kind: val`, and CLI dispatch.
