from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from .schema import ActionCandidateBatch, ModelConfig, StepBatch


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 512) -> None:
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_proj = num_proj
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        device = projections.device
        proj_dim = projections.size(-1)
        t = cast(torch.Tensor, self.t)
        phi = cast(torch.Tensor, self.phi)
        weights = cast(torch.Tensor, self.weights)
        random_basis = torch.randn(proj_dim, self.num_proj, device=device)
        random_basis = random_basis.div_(random_basis.norm(p=2, dim=0).clamp_min(1e-6))
        x_t = (projections @ random_basis).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * projections.size(-2)
        return statistic.mean()


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class StructuredStateEncoder(nn.Module):
    def __init__(
        self,
        *,
        observation_numeric_dim: int,
        observation_mask_dim: int,
        observation_cardinalities: list[int],
        d_state: int,
    ) -> None:
        super().__init__()
        numeric_input_dim = observation_numeric_dim + observation_mask_dim
        hidden_dim = max(d_state, 64)
        self.numeric_encoder = (
            MLPBlock(numeric_input_dim, hidden_dim, d_state) if numeric_input_dim > 0 else None
        )
        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, _embedding_dim(cardinality))
                for cardinality in observation_cardinalities
            ]
        )
        cat_dim = sum(
            cast(nn.Embedding, embedding).embedding_dim for embedding in self.categorical_embeddings
        )
        self.categorical_projection = (
            MLPBlock(cat_dim, hidden_dim, d_state) if cat_dim > 0 else None
        )
        combined_dim = (d_state if self.numeric_encoder is not None else 0) + (
            d_state if self.categorical_projection is not None else 0
        )
        self.output_projection = (
            MLPBlock(combined_dim, max(d_state, combined_dim), d_state)
            if combined_dim > 0
            else None
        )
        self.empty_token = nn.Parameter(torch.zeros(d_state))

    def forward(
        self,
        observation_numeric: torch.Tensor,
        observation_masks: torch.Tensor,
        observation_categorical: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.numeric_encoder is not None:
            numeric_input = torch.cat([observation_numeric, observation_masks], dim=-1)
            parts.append(self.numeric_encoder(numeric_input))
        if self.categorical_projection is not None:
            embeddings = [
                cast(nn.Embedding, embedding)(observation_categorical[..., index])
                for index, embedding in enumerate(self.categorical_embeddings)
            ]
            categorical_input = torch.cat(embeddings, dim=-1)
            parts.append(self.categorical_projection(categorical_input))
        if not parts:
            shape = observation_numeric.shape[:-1] + (self.empty_token.numel(),)
            return self.empty_token.view(1, 1, -1).expand(shape)
        if len(parts) == 1:
            return parts[0]
        output_projection = self.output_projection
        if output_projection is None:
            return parts[0]
        return output_projection(torch.cat(parts, dim=-1))


class StructuredActionEncoder(nn.Module):
    def __init__(
        self,
        *,
        action_numeric_dim: int,
        action_mask_dim: int,
        action_cardinalities: list[int],
        d_action: int,
    ) -> None:
        super().__init__()
        numeric_input_dim = action_numeric_dim + action_mask_dim
        hidden_dim = max(d_action, 64)
        self.numeric_encoder = (
            MLPBlock(numeric_input_dim, hidden_dim, d_action) if numeric_input_dim > 0 else None
        )
        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, _embedding_dim(cardinality))
                for cardinality in action_cardinalities
            ]
        )
        cat_dim = sum(
            cast(nn.Embedding, embedding).embedding_dim for embedding in self.categorical_embeddings
        )
        self.categorical_projection = (
            MLPBlock(cat_dim, hidden_dim, d_action) if cat_dim > 0 else None
        )
        combined_dim = (d_action if self.numeric_encoder is not None else 0) + (
            d_action if self.categorical_projection is not None else 0
        )
        self.output_projection = (
            MLPBlock(combined_dim, max(d_action, combined_dim), d_action)
            if combined_dim > 0
            else None
        )
        self.null_action = nn.Parameter(torch.zeros(d_action))

    def forward(
        self,
        action_numeric: torch.Tensor,
        action_masks: torch.Tensor,
        action_categorical: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.numeric_encoder is not None:
            numeric_input = torch.cat([action_numeric, action_masks], dim=-1)
            parts.append(self.numeric_encoder(numeric_input))
        if self.categorical_projection is not None:
            embeddings = [
                cast(nn.Embedding, embedding)(action_categorical[..., index])
                for index, embedding in enumerate(self.categorical_embeddings)
            ]
            categorical_input = torch.cat(embeddings, dim=-1)
            parts.append(self.categorical_projection(categorical_input))
        if not parts:
            shape = action_numeric.shape[:-1] + (self.null_action.numel(),)
            return self.null_action.view(1, 1, -1).expand(shape)
        if len(parts) == 1:
            encoded = parts[0]
        else:
            output_projection = self.output_projection
            if output_projection is None:
                encoded = parts[0]
            else:
                encoded = output_projection(torch.cat(parts, dim=-1))
        no_action_rows = (
            (action_numeric.abs().sum(dim=-1) + action_masks.abs().sum(dim=-1)) == 0
        ).float()
        if action_categorical.size(-1) > 0:
            no_action_rows = no_action_rows * (action_categorical.sum(dim=-1) == 0).float()
        null_action = self.null_action.view(1, 1, -1).expand_as(encoded)
        return encoded * (
            1.0 - no_action_rows.unsqueeze(-1)
        ) + null_action * no_action_rows.unsqueeze(-1)


class CausalPredictor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.position_embeddings = nn.Parameter(
            torch.randn(1, config.context_length, config.d_state)
        )
        self.input_projection = nn.Linear(config.d_state + config.d_action, config.d_state)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_state,
            nhead=config.heads,
            dropout=config.dropout,
            batch_first=True,
            dim_feedforward=config.d_state * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.output_projection = nn.Linear(config.d_state, config.d_state)

    def forward(self, state_latents: torch.Tensor, action_latents: torch.Tensor) -> torch.Tensor:
        time_steps = state_latents.size(1)
        inputs = torch.cat([state_latents, action_latents], dim=-1)
        hidden = self.input_projection(inputs)
        hidden = hidden + self.position_embeddings[:, :time_steps]
        causal_mask = torch.triu(
            torch.ones(time_steps, time_steps, device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.output_projection(hidden)


@dataclass
class ForwardPass:
    state_latents: torch.Tensor
    action_latents: torch.Tensor
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    prediction_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    total_loss: torch.Tensor


class StructuredStateJEPA(nn.Module):
    def __init__(
        self,
        *,
        config: ModelConfig,
        observation_numeric_dim: int,
        observation_mask_dim: int,
        observation_cardinalities: list[int],
        action_numeric_dim: int,
        action_mask_dim: int,
        action_cardinalities: list[int],
    ) -> None:
        super().__init__()
        self.config = config
        self.state_encoder = StructuredStateEncoder(
            observation_numeric_dim=observation_numeric_dim,
            observation_mask_dim=observation_mask_dim,
            observation_cardinalities=observation_cardinalities,
            d_state=config.d_state,
        )
        self.action_encoder = StructuredActionEncoder(
            action_numeric_dim=action_numeric_dim,
            action_mask_dim=action_mask_dim,
            action_cardinalities=action_cardinalities,
            d_action=config.d_action,
        )
        self.predictor = CausalPredictor(config)
        self.sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj)

    def encode_steps(self, batch: StepBatch) -> torch.Tensor:
        return self.state_encoder(
            batch.observation_numeric,
            batch.observation_masks,
            batch.observation_categorical,
        )

    def encode_actions(self, batch: StepBatch) -> torch.Tensor:
        return self.action_encoder(
            batch.action_numeric,
            batch.action_masks,
            batch.action_categorical,
        )

    def predict_next(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor) -> torch.Tensor:
        return self.predictor(z_ctx, a_ctx)

    def forward(self, batch: StepBatch) -> ForwardPass:
        state_latents = self.encode_steps(batch)
        action_latents = self.encode_actions(batch)
        predicted = self.predict_next(state_latents[:, :-1], action_latents[:, :-1])
        targets = state_latents[:, 1:]
        mask = batch.valid_mask[:, 1:].unsqueeze(-1)
        prediction_error = (predicted - targets).pow(2)
        prediction_loss = (prediction_error * mask).sum() / mask.sum().clamp_min(1.0)
        sigreg_loss = self.sigreg(state_latents.transpose(0, 1))
        total_loss = prediction_loss + self.config.sigreg_lambda * sigreg_loss
        return ForwardPass(
            state_latents=state_latents,
            action_latents=action_latents,
            predicted_latents=predicted,
            target_latents=targets,
            prediction_loss=prediction_loss,
            sigreg_loss=sigreg_loss,
            total_loss=total_loss,
        )

    @torch.no_grad()
    def surprise_score(self, batch: StepBatch) -> torch.Tensor:
        forward_pass = self.forward(batch)
        return (forward_pass.predicted_latents - forward_pass.target_latents).pow(2).mean(dim=-1)

    @torch.no_grad()
    def rollout(self, initial_steps: StepBatch, action_sequence: StepBatch) -> torch.Tensor:
        encoded_steps = self.encode_steps(initial_steps)
        encoded_actions = self.encode_actions(initial_steps)
        future_actions = self.action_encoder(
            action_sequence.action_numeric,
            action_sequence.action_masks,
            action_sequence.action_categorical,
        )

        history = encoded_steps.clone()
        action_history = encoded_actions.clone()
        predictions: list[torch.Tensor] = []
        for step_index in range(future_actions.size(1)):
            truncated_states = history[:, -self.config.context_length :]
            truncated_actions = action_history[:, -self.config.context_length :]
            predicted = self.predict_next(truncated_states, truncated_actions)[:, -1:]
            predictions.append(predicted)
            history = torch.cat([history, predicted], dim=1)
            action_history = torch.cat(
                [action_history, future_actions[:, step_index : step_index + 1]], dim=1
            )
        return (
            torch.cat(predictions, dim=1) if predictions else torch.zeros_like(encoded_steps[:, :0])
        )

    @torch.no_grad()
    def score_action_candidates(
        self,
        initial_steps: StepBatch,
        candidate_actions: ActionCandidateBatch,
        target_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sample_count, horizon, _ = candidate_actions.action_numeric.shape
        flat_numeric = candidate_actions.action_numeric.reshape(
            batch_size * sample_count, horizon, -1
        )
        flat_masks = candidate_actions.action_masks.reshape(batch_size * sample_count, horizon, -1)
        flat_categorical = candidate_actions.action_categorical.reshape(
            batch_size * sample_count, horizon, -1
        )
        repeated_initial = StepBatch(
            observation_numeric=initial_steps.observation_numeric.repeat_interleave(
                sample_count, dim=0
            ),
            observation_masks=initial_steps.observation_masks.repeat_interleave(
                sample_count, dim=0
            ),
            observation_categorical=initial_steps.observation_categorical.repeat_interleave(
                sample_count, dim=0
            ),
            action_numeric=initial_steps.action_numeric.repeat_interleave(sample_count, dim=0),
            action_masks=initial_steps.action_masks.repeat_interleave(sample_count, dim=0),
            action_categorical=initial_steps.action_categorical.repeat_interleave(
                sample_count, dim=0
            ),
            auxiliary_numeric_targets=initial_steps.auxiliary_numeric_targets.repeat_interleave(
                sample_count, dim=0
            ),
            valid_mask=initial_steps.valid_mask.repeat_interleave(sample_count, dim=0),
        )
        action_batch = StepBatch(
            observation_numeric=torch.zeros(
                (batch_size * sample_count, horizon, 0), device=flat_numeric.device
            ),
            observation_masks=torch.zeros(
                (batch_size * sample_count, horizon, 0), device=flat_numeric.device
            ),
            observation_categorical=torch.zeros(
                (batch_size * sample_count, horizon, 0),
                dtype=torch.long,
                device=flat_numeric.device,
            ),
            action_numeric=flat_numeric,
            action_masks=flat_masks,
            action_categorical=flat_categorical,
            auxiliary_numeric_targets=torch.zeros(
                (batch_size * sample_count, horizon, 0), device=flat_numeric.device
            ),
            valid_mask=torch.ones((batch_size * sample_count, horizon), device=flat_numeric.device),
        )
        rollout = self.rollout(repeated_initial, action_batch)
        terminal = rollout[:, -1]
        if target_latent is None:
            target_latent = self.encode_steps(initial_steps)[:, -1].repeat_interleave(
                sample_count, dim=0
            )
        else:
            target_latent = target_latent.repeat_interleave(sample_count, dim=0)
        costs = F.mse_loss(terminal, target_latent, reduction="none").mean(dim=-1)
        return costs.view(batch_size, sample_count)


def _embedding_dim(cardinality: int) -> int:
    if cardinality <= 4:
        return 4
    if cardinality <= 16:
        return 8
    return 16
