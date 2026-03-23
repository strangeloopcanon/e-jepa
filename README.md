# Structured-State JEPA

This project turns the LeWorldModel idea into a practical world model for structured business data.

It can:

- prepare wide step datasets from generic business time series
- prepare the same dataset shape from VEI workflow runs and playable runs
- prepare the same dataset shape from repeated VEI context captures
- train a compact JEPA that predicts the next latent state from the current state and action
- score surprise from prediction error and fit lightweight probes after training

## Why it exists

The original LeWorldModel paper learns from pixels. This repo keeps the same core idea, but swaps the image encoder for a structured-state encoder so we can use it on real business sequences and VEI enterprise simulations.

## The dataset format

Every prepared dataset writes two files:

- `steps.parquet`
- `schema.json`

Each row includes:

- `episode_id`
- `step_idx`
- `timestamp`
- `delta_t_s`
- `done`
- `split`
- `action_name`

Feature columns are prefixed so they stay easy to scan:

- `obs_num__...`
- `obs_cat__...`
- `act_num__...`
- `act_cat__...`
- `mask__...`
- `aux_num__...`
- `meta__...`

## Quick start

```bash
make setup

uv run structured-jepa prepare-timeseries \
  --input /path/to/business.csv \
  --output ./out/business_dataset \
  --entity-column account_id \
  --timestamp-column event_time

uv run structured-jepa train \
  --dataset ./out/business_dataset \
  --output ./out/checkpoints/business_model
```

## VEI examples

Prepare data from VEI runs:

```bash
uv run structured-jepa prepare-vei-runs \
  --workspace-root /path/to/vei/workspace \
  --output ./out/vei_runs_dataset
```

Prepare data from VEI context snapshots:

```bash
uv run structured-jepa prepare-vei-context \
  --snapshot-glob "/path/to/context/*.json" \
  --output ./out/vei_context_dataset
```

## Verification

```bash
make check
make test
```
