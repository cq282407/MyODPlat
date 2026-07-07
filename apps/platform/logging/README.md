# apps/platform/logging

## Current Status

Runtime log files are written here by `odp-init` and future platform commands.

## Purpose

Logs are app-private runtime artifacts. They are ignored by git except for this
README.

## Future Plan

Future production deployments may replace file logs with stdout/stderr and a
central log collector.
