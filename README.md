# E-JEPA

E-JEPA is a JEPA world model for enterprise data.

Instead of learning from pixels, it learns from structured business state and the decisions taken in that state.

## What It Is

E-JEPA is for settings where the world already looks like tables, counts, statuses, events, and actions.

Examples:

- business time series
- enterprise operations data
- VEI workflow runs
- VEI playable missions
- repeated enterprise context snapshots from systems like Slack, Jira, Google, and Okta

The model's job is simple:

- read the recent state
- read the recent actions
- predict the next state in a compact latent space

It does not currently predict the next action. Actions are inputs that help it predict what happens next.

## How It Relates To LeWorldModel

This is not a direct modification of [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm).

LeWorldModel is the pixel version. Its own README says that repository is built on top of `stable-worldmodel` and `stable-pretraining`, with that repo focused on the model and objective. E-JEPA keeps the same high-level JEPA recipe, but reworks the front end, data format, and training interface for enterprise state and decision data instead of image trajectories.

In short:

- LeWorldModel: pixels in, latent future prediction out
- E-JEPA: structured enterprise state and action in, latent future prediction out

## What Changed

Compared with the pixel setting, E-JEPA changes four main things:

- the image encoder is replaced with a structured-state encoder for numeric, categorical, and missing-value features
- actions are represented explicitly as business controls or VEI tool choices
- raw business and VEI data are converted into one common step format: `steps.parquet` plus `schema.json`
- evaluation is geared toward enterprise use cases: next-state prediction, surprise scoring, and simple readouts of business quantities

## How Training Works

Training is intentionally simple.

1. Prepare sequences into a common step dataset.
2. Normalize numeric fields on the training split only.
3. Embed categorical fields and action fields.
4. Encode each step into a compact latent state.
5. Feed the last 16 latent states and aligned actions into a causal predictor.
6. Train the predictor to match the next latent state.
7. Add the same Gaussian regularizer idea used in LeWorldModel to keep the latent space well-behaved.

So the core learning target is:

- current state plus action history -> next latent state

After training, you can use that predictor for:

- next-state forecasting
- surprise detection from prediction error
- lightweight probes for business quantities
- future planning work, using candidate action scoring

## Does Training Work?

Yes, as a first working version.

- the full project checks and tests pass
- live VEI workflow and playable exports both worked against the updated VEI code
- on synthetic business data, the model beat a simple persistence baseline
- on an action-driven synthetic benchmark, including actions improved next-state prediction over leaving actions out

What it is good for today:

- learning compact state dynamics over enterprise data
- using actions to improve next-state prediction
- flagging unusual transitions

What it is not yet:

- a finished planner
- a next-action generator
- a large-scale real-business benchmark result

## What It Can Do

- prepare wide step datasets from generic business time series
- prepare the same dataset shape from VEI workflow runs and playable runs
- prepare the same dataset shape from repeated VEI context captures
- train a compact JEPA that predicts the next latent state from the current state and action
- score surprise from prediction error and fit lightweight probes after training

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

uv run e-jepa prepare-timeseries \
  --input /path/to/business.csv \
  --output ./out/business_dataset \
  --entity-column account_id \
  --timestamp-column event_time

uv run e-jepa train \
  --dataset ./out/business_dataset \
  --output ./out/checkpoints/business_model
```

## VEI examples

Prepare data from VEI runs:

```bash
uv run e-jepa prepare-vei-runs \
  --workspace-root /path/to/vei/workspace \
  --output ./out/vei_runs_dataset
```

Prepare data from VEI context snapshots:

```bash
uv run e-jepa prepare-vei-context \
  --snapshot-glob "/path/to/context/*.json" \
  --output ./out/vei_context_dataset
```

## Verification

```bash
make check
make test
```
