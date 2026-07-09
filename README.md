# ODPlatform

## Current Status

This repository contains the current offline ODPlatform workflow under
`apps/platform`: project initialization, reset, data conversion, data
validation, runtime config generation, YOLO training, YOLO model
evaluation (`odp-val`), and YOLO model inference (`odp-infer`).

## Purpose

ODPlatform is a target-detection platform skeleton. The current implementation
focuses on reusable engineering infrastructure for the offline pipeline, while
future web and desktop applications will build on the same core.

## Available Commands

- `odp-init`
- `odp-reset`
- `odp-transform`
- `odp-validate`
- `odp-gen-config`
- `odp-train`
- `odp-val`
- `odp-infer`

## Operation Guides

- D7 evaluation guide: [docs/guides/D7-evaluation-guide.md](docs/guides/D7-evaluation-guide.md)
- D8 inference guide: [docs/guides/D8-inference-guide.md](docs/guides/D8-inference-guide.md)
- GitHub publish guide: [docs/guides/publish-to-github.md](docs/guides/publish-to-github.md)

## Future Plan

Future milestones will add richer reporting, acceleration and deployment
recipes, and additional app entry points such as web and desktop clients.
