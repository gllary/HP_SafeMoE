from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Stage2ProposalConfig:
    task_names: tuple[str, ...]
    expert_ids: tuple[str, ...]
    hidden_dims: dict[str, int]
    anchor_indices: tuple[int, ...]
    token_dim: int
    task_dim: int
    residual_hidden: int
    residual_rank: int
    dropout: float
    attention_top_k: int
    residual_limit: float
    route_temperature: float
    use_cross_expert_input: bool
    use_data_branch: bool
    use_relation_branch: bool
    use_learned_task_source_relation: bool
    use_shared_low_rank_basis: bool
    use_task_private_correction: bool
    use_sample_gate: bool


@dataclass
class Stage2ProposalOutput:
    raw_standard: torch.Tensor
    data_standard: torch.Tensor
    relation_standard: torch.Tensor
    data_residual: torch.Tensor
    relation_residual: torch.Tensor
    route_probabilities: torch.Tensor
    route_one_hot: torch.Tensor
    data_attention: torch.Tensor
    relation_attention: torch.Tensor
    reliability: torch.Tensor
    learned_relation_scale: torch.Tensor


def reliability_features(
    *,
    anchor_standard: torch.Tensor,
    prediction_standard: dict[str, torch.Tensor],
    uncertainty_standard: dict[str, torch.Tensor],
    available: torch.Tensor,
    expert_ids: tuple[str, ...],
) -> torch.Tensor:
    predictions = torch.stack([prediction_standard[name] for name in expert_ids], dim=1)
    uncertainties = torch.stack([uncertainty_standard[name] for name in expert_ids], dim=1)
    mask = available.to(predictions.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    mean_prediction = (predictions * mask).sum(dim=1) / count
    centered = (predictions - mean_prediction[:, None]) * mask
    disagreement = torch.sqrt(centered.square().sum(dim=1) / count + 1.0e-8)
    mean_uncertainty = (uncertainties * mask).sum(dim=1) / count
    maximum_uncertainty = (
        uncertainties.masked_fill(~available, torch.finfo(uncertainties.dtype).min).max(dim=1).values
    )
    maximum_prediction = predictions.abs().masked_fill(~available, 0.0).max(dim=1).values
    return torch.stack(
        [
            anchor_standard,
            anchor_standard.abs(),
            count / float(len(expert_ids)),
            disagreement,
            (anchor_standard - mean_prediction).abs(),
            mean_uncertainty,
            maximum_uncertainty,
            maximum_prediction,
        ],
        dim=1,
    )


class CrossExpertProposalModel(nn.Module):
    """Learn an ungated Data/Relation proposal from one outer-training subset.

    The target-task expert is represented by the Stage-1 anchor, while both
    proposal branches attend to the other frozen experts. The Relation branch
    learns a task-source matrix from outer-training data. A downstream controller
    learns safety from held-out inner-OOF proposals.
    """

    RELIABILITY_DIM = 8
    ANCHOR_ROUTE = 0
    DATA_ROUTE = 1
    RELATION_ROUTE = 2

    def __init__(self, config: Stage2ProposalConfig) -> None:
        super().__init__()
        self.config = config
        tasks, experts = len(config.task_names), len(config.expert_ids)
        if len(config.anchor_indices) != tasks:
            raise ValueError("Anchor indices must align with task names")
        if set(config.hidden_dims) != set(config.expert_ids):
            raise ValueError("Hidden dimensions must cover exactly the expert IDs")
        if not config.use_data_branch and not config.use_relation_branch:
            raise ValueError("At least one cross-expert proposal branch is required")
        if not config.use_shared_low_rank_basis and not config.use_task_private_correction:
            raise ValueError("At least one residual output component is required")
        if not 1 <= config.attention_top_k <= experts:
            raise ValueError("attention_top_k is outside the expert range")
        if config.route_temperature <= 0.0:
            raise ValueError("route_temperature must be positive")

        self.register_buffer("anchor_indices", torch.tensor(config.anchor_indices, dtype=torch.long))
        self.task_embedding = nn.Embedding(tasks, config.task_dim)
        self.expert_embedding = nn.Embedding(experts, config.token_dim)
        self.hidden_adapters = nn.ModuleDict(
            {
                expert_id: nn.Sequential(
                    nn.LayerNorm(config.hidden_dims[expert_id]),
                    nn.Linear(config.hidden_dims[expert_id], config.token_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.token_dim, config.token_dim),
                )
                for expert_id in config.expert_ids
            }
        )
        self.scalar_adapter = nn.Sequential(
            nn.Linear(2, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        attention_input = config.token_dim + config.task_dim + self.RELIABILITY_DIM
        self.data_attention_network = nn.Sequential(
            nn.LayerNorm(attention_input),
            nn.Linear(attention_input, config.residual_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.residual_hidden, 1),
        )
        self.relation_attention_network = nn.Sequential(
            nn.LayerNorm(attention_input + 1),
            nn.Linear(attention_input + 1, config.residual_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.residual_hidden, 1),
        )
        self.data_uncertainty_penalty = nn.Parameter(torch.tensor(-1.0))
        self.relation_uncertainty_penalty = nn.Parameter(torch.tensor(-1.0))
        if config.use_learned_task_source_relation:
            # The complete relation matrix starts neutral and is learned only
            # from the current official outer-train subset.
            self.task_source_relation = nn.Parameter(torch.zeros(tasks, experts))
            self.learned_relation_strength = nn.Parameter(torch.tensor(-1.0))
        else:
            self.register_parameter("task_source_relation", None)
            self.register_parameter("learned_relation_strength", None)

        self.branch_feature_dim = 4 * config.token_dim + config.task_dim + self.RELIABILITY_DIM + 1
        if config.use_shared_low_rank_basis:
            self.residual_basis = nn.Sequential(
                nn.LayerNorm(self.branch_feature_dim),
                nn.Linear(self.branch_feature_dim, config.residual_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.residual_hidden, config.residual_rank),
            )
            self.task_branch_coefficients = nn.Embedding(tasks * 2, config.residual_rank)
            nn.init.normal_(self.task_branch_coefficients.weight, mean=0.0, std=0.05)
            nn.init.zeros_(self.residual_basis[-1].weight)
            nn.init.zeros_(self.residual_basis[-1].bias)
        else:
            self.residual_basis = None
            self.task_branch_coefficients = None

        if config.use_task_private_correction:
            self.private_residual = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.branch_feature_dim),
                        nn.Linear(self.branch_feature_dim, config.residual_hidden),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.residual_hidden, 1),
                    )
                    for _ in range(tasks * 2)
                ]
            )
            for head in self.private_residual:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        else:
            self.private_residual = None

        route_input = 4 * config.token_dim + config.task_dim + self.RELIABILITY_DIM + 2
        self.sample_route = nn.Sequential(
            nn.LayerNorm(route_input),
            nn.Linear(route_input, config.residual_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.residual_hidden, 3),
        )
        nn.init.zeros_(self.sample_route[-1].weight)
        initial_route = [-1.0, 0.0, 0.0]
        with torch.no_grad():
            self.sample_route[-1].bias.copy_(torch.tensor(initial_route))

        self.task_branch_logits = nn.Embedding(tasks, 2)
        nn.init.zeros_(self.task_branch_logits.weight)

    def _tokens(
        self,
        *,
        hidden: dict[str, torch.Tensor],
        prediction_standard: dict[str, torch.Tensor],
        uncertainty_standard: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, uncertainty = [], []
        for index, expert_id in enumerate(self.config.expert_ids):
            scalar = torch.stack([prediction_standard[expert_id], uncertainty_standard[expert_id]], dim=1)
            tokens.append(
                self.hidden_adapters[expert_id](hidden[expert_id])
                + self.scalar_adapter(scalar)
                + self.expert_embedding.weight[index]
            )
            uncertainty.append(uncertainty_standard[expert_id])
        return torch.stack(tokens, dim=1), torch.stack(uncertainty, dim=1)

    def _cross_mask(
        self, task_index: torch.Tensor, available: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.arange(len(task_index), device=task_index.device)
        mask = available.clone()
        if not self.config.use_cross_expert_input:
            mask[:] = False
            mask[rows, self.anchor_indices[task_index]] = True
            return mask, available[rows, self.anchor_indices[task_index]]
        mask[rows, self.anchor_indices[task_index]] = False
        cross_available = mask.any(dim=1)
        if torch.any(~cross_available):
            mask[~cross_available, self.anchor_indices[task_index[~cross_available]]] = True
        return mask, cross_available

    def _sparse_attention(
        self,
        *,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        unavailable = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, unavailable)
        top_k = min(self.config.attention_top_k, logits.shape[1])
        top_values, top_indices = torch.topk(logits, k=top_k, dim=1)
        top_valid = torch.gather(mask, 1, top_indices)
        top_values = top_values.masked_fill(~top_valid, unavailable)
        sparse = torch.full_like(logits, unavailable)
        sparse.scatter_(1, top_indices, top_values)
        return torch.softmax(sparse.float(), dim=1).to(logits.dtype)

    def _attention_contexts(
        self,
        *,
        task_index: torch.Tensor,
        task: torch.Tensor,
        token_matrix: torch.Tensor,
        uncertainty: torch.Tensor,
        reliability: torch.Tensor,
        cross_mask: torch.Tensor,
        cross_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expanded_task = task[:, None, :].expand(-1, token_matrix.shape[1], -1)
        expanded_reliability = reliability[:, None, :].expand(-1, token_matrix.shape[1], -1)
        common = torch.cat([token_matrix, expanded_task, expanded_reliability], dim=2)

        data_logits = self.data_attention_network(common).squeeze(2)
        data_logits = data_logits - F.softplus(self.data_uncertainty_penalty) * uncertainty
        data_attention = self._sparse_attention(logits=data_logits, mask=cross_mask)
        data_attention = data_attention * cross_available[:, None].to(data_attention.dtype)
        data_context = torch.sum(data_attention[:, :, None] * token_matrix, dim=1)

        if self.config.use_learned_task_source_relation:
            assert self.task_source_relation is not None
            assert self.learned_relation_strength is not None
            relation = torch.tanh(self.task_source_relation[task_index])
            relation_scale = F.softplus(self.learned_relation_strength)
        else:
            relation = torch.zeros(
                (len(task_index), token_matrix.shape[1]),
                dtype=token_matrix.dtype,
                device=token_matrix.device,
            )
            relation_scale = token_matrix.new_zeros(())
        relation_logits = self.relation_attention_network(
            torch.cat([common, relation[:, :, None]], dim=2)
        ).squeeze(2)
        relation_logits = relation_logits - F.softplus(self.relation_uncertainty_penalty) * uncertainty
        relation_logits = relation_logits + relation_scale * relation
        relation_attention = self._sparse_attention(logits=relation_logits, mask=cross_mask)
        relation_attention = relation_attention * cross_available[:, None].to(relation_attention.dtype)
        relation_context = torch.sum(relation_attention[:, :, None] * token_matrix, dim=1)
        return (
            data_context,
            relation_context,
            data_attention,
            relation_attention,
            relation_scale,
        )

    def _branch_features(
        self,
        *,
        branch_token: torch.Tensor,
        anchor_token: torch.Tensor,
        task: torch.Tensor,
        reliability: torch.Tensor,
        anchor_standard: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                branch_token,
                anchor_token,
                branch_token - anchor_token,
                branch_token * anchor_token,
                task,
                reliability,
                anchor_standard[:, None],
            ],
            dim=1,
        )

    def _branch_residual(self, features: torch.Tensor, task_index: torch.Tensor, branch: int) -> torch.Tensor:
        head_index = task_index * 2 + int(branch)
        components: list[torch.Tensor] = []
        if self.config.use_shared_low_rank_basis:
            assert self.residual_basis is not None and self.task_branch_coefficients is not None
            basis = self.residual_basis(features)
            coefficients = self.task_branch_coefficients(head_index)
            components.append(
                torch.sum(basis * coefficients, dim=1) / math.sqrt(float(self.config.residual_rank))
            )
        if self.config.use_task_private_correction:
            assert self.private_residual is not None
            private = torch.empty(len(task_index), dtype=features.dtype, device=features.device)
            for index, head in enumerate(self.private_residual):
                selected = head_index == index
                if torch.any(selected):
                    private[selected] = head(features[selected]).squeeze(1)
            components.append(private)
        raw = torch.stack(components, dim=0).sum(dim=0) / math.sqrt(float(len(components)))
        limit = float(self.config.residual_limit)
        return limit * torch.tanh(raw / limit)

    def _route(
        self,
        *,
        logits: torch.Tensor,
        cross_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allowed = torch.ones_like(logits, dtype=torch.bool)
        # Cross-source inputs route through the Data or Relation branches. The
        # anchor route handles unavailable cross-source input, and the OOF safety
        # controller determines deployment acceptance.
        allowed[:, self.ANCHOR_ROUTE] = ~cross_available
        allowed[:, self.DATA_ROUTE] = cross_available & self.config.use_data_branch
        allowed[:, self.RELATION_ROUTE] = cross_available & self.config.use_relation_branch
        masked = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(masked.float() / self.config.route_temperature, dim=1).to(logits.dtype)
        hard = F.one_hot(probabilities.argmax(dim=1), num_classes=3).to(logits.dtype)
        route = hard + probabilities - probabilities.detach() if self.training else hard
        return route, probabilities

    def forward(
        self,
        *,
        task_index: torch.Tensor,
        anchor_standard: torch.Tensor,
        hidden: dict[str, torch.Tensor],
        prediction_standard: dict[str, torch.Tensor],
        uncertainty_standard: dict[str, torch.Tensor],
        available: torch.Tensor,
    ) -> Stage2ProposalOutput:
        if available.ndim != 2 or available.shape[1] != len(self.config.expert_ids):
            raise ValueError("available must have shape [batch, experts]")
        reliability = reliability_features(
            anchor_standard=anchor_standard,
            prediction_standard=prediction_standard,
            uncertainty_standard=uncertainty_standard,
            available=available,
            expert_ids=self.config.expert_ids,
        )
        task = self.task_embedding(task_index)
        token_matrix, uncertainty = self._tokens(
            hidden=hidden,
            prediction_standard=prediction_standard,
            uncertainty_standard=uncertainty_standard,
        )
        rows = torch.arange(len(task_index), device=task_index.device)
        anchor_token = token_matrix[rows, self.anchor_indices[task_index]]
        cross_mask, cross_available = self._cross_mask(task_index, available)
        data_token, relation_token, data_attention, relation_attention, relation_scale = (
            self._attention_contexts(
                task_index=task_index,
                task=task,
                token_matrix=token_matrix,
                uncertainty=uncertainty,
                reliability=reliability,
                cross_mask=cross_mask,
                cross_available=cross_available,
            )
        )
        data_features = self._branch_features(
            branch_token=data_token,
            anchor_token=anchor_token,
            task=task,
            reliability=reliability,
            anchor_standard=anchor_standard,
        )
        relation_features = self._branch_features(
            branch_token=relation_token,
            anchor_token=anchor_token,
            task=task,
            reliability=reliability,
            anchor_standard=anchor_standard,
        )
        data_residual = self._branch_residual(data_features, task_index, branch=0)
        relation_residual = self._branch_residual(relation_features, task_index, branch=1)
        if not self.config.use_data_branch:
            data_residual = torch.zeros_like(data_residual)
        if not self.config.use_relation_branch:
            relation_residual = torch.zeros_like(relation_residual)
        data_residual = torch.where(cross_available, data_residual, torch.zeros_like(data_residual))
        relation_residual = torch.where(cross_available, relation_residual, torch.zeros_like(relation_residual))
        data_standard = anchor_standard + data_residual
        relation_standard = anchor_standard + relation_residual

        if self.config.use_sample_gate:
            route_features = torch.cat(
                [
                    anchor_token,
                    data_token,
                    relation_token,
                    data_token - relation_token,
                    task,
                    reliability,
                    data_residual[:, None],
                    relation_residual[:, None],
                ],
                dim=1,
            )
            route, probabilities = self._route(
                logits=self.sample_route(route_features), cross_available=cross_available
            )
            candidates = torch.stack([anchor_standard, data_standard, relation_standard], dim=1)
            proposal = torch.sum(route * candidates, dim=1)
        else:
            branch_logits = self.task_branch_logits(task_index)
            branch_allowed = torch.tensor(
                [self.config.use_data_branch, self.config.use_relation_branch],
                dtype=torch.bool,
                device=task_index.device,
            )
            branch_logits = branch_logits.masked_fill(
                ~branch_allowed[None, :], torch.finfo(branch_logits.dtype).min
            )
            branch_probability = torch.softmax(branch_logits.float(), dim=1).to(branch_logits.dtype)
            proposal = branch_probability[:, 0] * data_standard + branch_probability[:, 1] * relation_standard
            probabilities = torch.cat(
                [torch.zeros(len(task_index), 1, device=task_index.device), branch_probability],
                dim=1,
            )
            route = F.one_hot(probabilities.argmax(dim=1), num_classes=3).to(probabilities.dtype)

        raw_standard = proposal
        route_hard = F.one_hot(route.argmax(dim=1), num_classes=3).to(route.dtype)
        return Stage2ProposalOutput(
            raw_standard=raw_standard,
            data_standard=data_standard,
            relation_standard=relation_standard,
            data_residual=data_residual,
            relation_residual=relation_residual,
            route_probabilities=probabilities,
            route_one_hot=route_hard if not self.training else route,
            data_attention=data_attention,
            relation_attention=relation_attention,
            reliability=reliability,
            learned_relation_scale=relation_scale,
        )
