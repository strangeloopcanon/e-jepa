# Fast-WAM-Inspired State V2 Experiment

Status: future experiment, not part of the current E-JEPA baseline

## Why This Exists

The current E-JEPA baseline works by turning enterprise state into one flat step table, then learning next-state dynamics from recent state and action history.

That is a good first version, but it likely throws away too much structure.

The Fast-WAM paper suggests a useful principle for us:

- keep inference simple
- put more effort into training-time representation learning

In other words, we should not rush into expensive test-time imagination machinery. A better next step is to make the enterprise state representation richer and more structured while keeping test-time use lightweight.

## Hypothesis

E-JEPA will improve more from a better structured state representation than from adding heavier test-time rollout or explicit imagined future generation.

## Main Idea

Keep the current basic contract:

- recent state
- recent actions
- predict the next state

But change the state representation from one wide flat summary into a typed, structured representation.

## Proposed Changes

### 1. Move From Flat Summaries To Typed Tokens

Instead of one flattened vector per step, represent enterprise state as a set of typed tokens, for example:

- global org token
- provider tokens for Slack, Jira, Google, Okta
- object tokens for channels, issues, users, groups, documents, incidents
- pending-event tokens
- action target tokens

Each token should carry:

- type
- key status fields
- small numeric summaries
- freshness
- missingness

### 2. Preserve Relations

Add lightweight relation structure instead of flattening everything away.

Examples:

- issue belongs to project
- user belongs to org unit
- action touched object
- event refers to object

The first version does not need a full graph neural network. It is enough to keep relation ids or typed links that can be embedded and attended over.

### 3. Separate Static State From Changing State

Do not represent stable setup data and live operational state the same way.

Examples:

- mostly static: org structure, project catalog, group catalog
- dynamic: counts, statuses, unread state, pending events, tool outcomes

This should reduce noise and make transition learning cleaner.

### 4. Keep Event Tokens Between Snapshots

The current baseline is snapshot-heavy.

Add explicit event tokens for things like:

- ticket opened or closed
- user suspended
- message sent
- document shared
- tool call completed

This should help the model learn what changed, not just what the world looked like before and after.

### 5. Add A Separate Decision Head

Do not make the world model itself do everything.

Train the world model to learn state dynamics. Then add a small decision head that reads the learned world state and predicts:

- action scores
- action categories
- action chunks

This follows the general lesson that the world-modeling objective may matter most during training, while inference should stay direct and cheap.

## Minimum Useful Experiment

Run a controlled comparison with three variants:

1. Current baseline
2. Baseline plus typed state tokens
3. Baseline plus typed state tokens and event tokens

Keep inference simple in all three.

Do not add explicit multi-step imagined future generation in this experiment.

## Datasets To Use

Use one synthetic benchmark and one real-ish benchmark.

Synthetic:

- action-driven business simulator where controls clearly affect next-state dynamics

Real-ish:

- VEI workflow runs
- VEI playable missions
- repeated VEI context snapshots

## What To Measure

Primary:

- next-state prediction quality
- surprise detection quality
- action-conditioned improvement over no-action ablation

Secondary:

- linear probe quality on business quantities
- robustness to missing data
- inference latency

## Success Criteria

Call the experiment successful if the structured variants:

- improve next-state prediction over the current flat baseline
- improve surprise separation on perturbed transitions
- keep inference near the current cost profile

## What Not To Do In This Experiment

- do not add expensive test-time imagination loops
- do not add a large planner first
- do not chase maximum model size before representation quality is tested

## If It Works

The next step would be:

- keep the structured world encoder
- add a stronger decision head on top of the learned state
- test candidate-action scoring for simple planning tasks

## References

- Fast-WAM: https://arxiv.org/pdf/2603.16666
- LeWorldModel paper: https://arxiv.org/pdf/2603.19312v1
- LeWorldModel project: https://le-wm.github.io/
