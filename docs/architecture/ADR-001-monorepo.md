# ADR-001: Adopt Monorepo with apps Layout

## Status

Accepted.

## Context

ODPlatform needs a structure that supports one current app (`platform`) and
future apps such as web backend, web frontend, and desktop clients. It also
needs shared data, models, runs, documentation, and consistent tooling.

## Decision

Use a single repository with an `apps/` directory for app-level deliverables.
The current engine lives in `apps/platform`. Shared assets live at repository
root, while app-private assets live inside each app.

## Consequences

Developers can install `apps/platform` once in editable mode and import
`od_platform` from future apps in the same Python environment. The repository
keeps shared tooling configuration at root while each installable app owns its
own package metadata and dependencies.
