# apps/platform

## Current Status

`platform` is the current ODPlatform core app. It provides the installable
Python package `od_platform` and the offline workflow commands for
initialization, reset, data transformation, data validation, config generation,
training, evaluation, and inference.

## Purpose

This app owns the reusable offline ML workflow infrastructure: data pipeline,
quality gate, runtime configuration, training orchestration, model evaluation,
and model inference.

## Implemented Commands

- `odp-init`
- `odp-reset`
- `odp-transform`
- `odp-validate`
- `odp-gen-config`
- `odp-train`
- `odp-val`
- `odp-infer`

## Related Guides

- [../../docs/guides/D7-evaluation-guide.md](../../docs/guides/D7-evaluation-guide.md)
- [../../docs/guides/D8-inference-guide.md](../../docs/guides/D8-inference-guide.md)
- [../../docs/guides/publish-to-github.md](../../docs/guides/publish-to-github.md)

## Future Plan

Later stages will add acceleration and deployment playbooks, richer reporting,
and supporting app entry points such as web and desktop clients.
