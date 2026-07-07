# apps/platform

## Current Status

`platform` is the first ODPlatform app. It provides the installable Python
package `od_platform` and the `odp-init` command.

## Purpose

This app owns the training and offline inference engine infrastructure. At this
stage it contains only common utilities and the initialization command.

## Future Plan

Later stages will add data pipeline, training, evaluation, and inference
modules under `src/od_platform/`.
