from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    shape = (size, *values.shape[1:])
    output = values.new_zeros(shape)
    output.index_add_(0, index, values)
    return output


def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    total = scatter_sum(values, index, size)
    count = scatter_sum(values.new_ones((len(values), 1)), index, size).clamp_min_(1.0)
    return total / count


def segment_softmax(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Stable softmax over sparse incoming-edge segments using stock PyTorch."""

    if values.ndim == 1:
        values = values[:, None]
    maximum = values.new_full((size, values.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, index[:, None].expand_as(values), values, reduce="amax", include_self=True)
    exponent = torch.exp(values - maximum[index])
    denominator = scatter_sum(exponent, index, size).clamp_min_(1e-12)
    return exponent / denominator[index]


def gaussian_rbf(distance: torch.Tensor, n_rbf: int, cutoff: float) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, n_rbf, device=distance.device, dtype=distance.dtype)
    width = max(cutoff / max(n_rbf - 1, 1), 1e-3)
    return torch.exp(-0.5 * ((distance[:, None] - centers[None, :]) / width) ** 2)


@dataclass
class CrystalBatch:
    atomic_numbers: torch.Tensor
    atom_features: torch.Tensor
    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_vector: torch.Tensor
    node_batch: torch.Tensor
    graph_features: torch.Tensor
    line_index: torch.Tensor
    line_cosine: torch.Tensor
    targets: torch.Tensor | None = None
    positions: torch.Tensor | None = None

    @property
    def n_graphs(self) -> int:
        return int(self.graph_features.shape[0])

    def to(self, device: torch.device) -> CrystalBatch:
        return CrystalBatch(
            atomic_numbers=self.atomic_numbers.to(device),
            atom_features=self.atom_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_distance=self.edge_distance.to(device),
            edge_vector=self.edge_vector.to(device),
            node_batch=self.node_batch.to(device),
            graph_features=self.graph_features.to(device),
            line_index=self.line_index.to(device),
            line_cosine=self.line_cosine.to(device),
            targets=None if self.targets is None else self.targets.to(device),
            positions=None if self.positions is None else self.positions.to(device),
        )


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, width * 2)
        self.contract = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.expand(self.norm(inputs))
        value, gate = hidden.chunk(2, dim=-1)
        return inputs + self.dropout(self.contract(torch.nn.functional.silu(value) * torch.sigmoid(gate)))


class NativeDescriptorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 384,
        latent_dim: int = 256,
        layers: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, dropout) for _ in range(layers)])
        self.latent = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim), nn.SiLU())
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input(features)
        for block in self.blocks:
            hidden = block(hidden)
        latent = self.latent(hidden)
        return self.head(latent).reshape(-1), latent


class EdgeUpdate(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        source, target = edge_index
        delta = self.message(torch.cat([nodes[source], nodes[target], edges], dim=-1))
        return self.norm(edges + delta)


class NodeUpdate(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 2, width)
        )
        self.update = nn.Sequential(
            nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 2, width)
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        source, target = edge_index
        messages = self.message(torch.cat([nodes[source], edges], dim=-1))
        aggregated = scatter_mean(messages, target, len(nodes))
        return self.norm(nodes + self.update(torch.cat([nodes, aggregated], dim=-1)))


class SparseAttentionUpdate(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.edge_bias = nn.Linear(width, heads)
        self.output = nn.Sequential(nn.Linear(width, width), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        source, target = edge_index
        q = self.query(nodes[target]).reshape(-1, self.heads, self.head_dim)
        k = self.key(nodes[source]).reshape(-1, self.heads, self.head_dim)
        v = self.value(nodes[source]).reshape(-1, self.heads, self.head_dim)
        score = (q * k).sum(-1) / self.head_dim**0.5 + self.edge_bias(edges)
        weights = segment_softmax(score, target, len(nodes))
        weighted = (weights[..., None] * v).reshape(-1, self.heads * self.head_dim)
        aggregated = scatter_sum(weighted, target, len(nodes))
        return self.norm(nodes + self.output(aggregated))


class NativeCrystalModel(nn.Module):
    """One dependency-free crystal backbone with architecture-specific message blocks."""

    def __init__(
        self,
        architecture: str,
        *,
        hidden_dim: int = 192,
        latent_dim: int = 256,
        layers: int = 4,
        rbf_dim: int = 64,
        cutoff: float = 8.0,
        heads: int = 6,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        if architecture not in {"edge_gnn", "alignn", "matformer", "cogn"}:
            raise ValueError(architecture)
        self.architecture = architecture
        self.rbf_dim = rbf_dim
        self.cutoff = cutoff
        self.atom_embedding = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.atom_property_projection = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU())
        self.edge_embedding = nn.Sequential(nn.Linear(rbf_dim + 3, hidden_dim), nn.SiLU())
        self.graph_embedding = nn.Sequential(nn.Linear(10, hidden_dim), nn.SiLU())
        self.edge_updates = nn.ModuleList([EdgeUpdate(hidden_dim, dropout) for _ in range(layers)])
        if architecture == "matformer":
            self.node_updates = nn.ModuleList(
                [SparseAttentionUpdate(hidden_dim, heads, dropout) for _ in range(layers)]
            )
        else:
            self.node_updates = nn.ModuleList([NodeUpdate(hidden_dim, dropout) for _ in range(layers)])
        if architecture == "alignn":
            self.angle_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2 + 16, hidden_dim * 2),
                        nn.SiLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    )
                    for _ in range(layers)
                ]
            )
            self.angle_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        if architecture == "cogn":
            self.state_updates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim * 2),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    )
                    for _ in range(layers)
                ]
            )
            self.state_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(latent_dim, 1)

    @staticmethod
    def _angle_rbf(cosine: torch.Tensor, width: int = 16) -> torch.Tensor:
        centers = torch.linspace(-1.0, 1.0, width, device=cosine.device, dtype=cosine.dtype)
        return torch.exp(-12.0 * (cosine[:, None] - centers[None, :]) ** 2)

    def forward(self, batch: CrystalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.atom_embedding(batch.atomic_numbers.clamp(0, 118)) + self.atom_property_projection(
            batch.atom_features
        )
        edge_input = torch.cat(
            [gaussian_rbf(batch.edge_distance, self.rbf_dim, self.cutoff), batch.edge_vector], dim=-1
        )
        edges = self.edge_embedding(edge_input)
        state = self.graph_embedding(batch.graph_features)
        edge_graph = batch.node_batch[batch.edge_index[0]]
        for layer, (edge_update, node_update) in enumerate(
            zip(self.edge_updates, self.node_updates, strict=True)
        ):
            if self.architecture == "alignn" and batch.line_index.shape[1] > 0:
                line_source, line_target = batch.line_index
                angular = self.angle_embeddings[layer](
                    torch.cat(
                        [edges[line_source], edges[line_target], self._angle_rbf(batch.line_cosine)], dim=-1
                    )
                )
                edges = self.angle_norms[layer](
                    edges + scatter_mean(angular, line_target, len(edges))
                )
            if self.architecture == "cogn":
                edge_state = state[edge_graph]
                edges = edges + edge_state
            edges = edge_update(nodes, edges, batch.edge_index)
            nodes = node_update(nodes, edges, batch.edge_index)
            if self.architecture == "cogn":
                node_pool = scatter_mean(nodes, batch.node_batch, batch.n_graphs)
                edge_pool = scatter_mean(edges, edge_graph, batch.n_graphs)
                state = self.state_norms[layer](
                    state + self.state_updates[layer](torch.cat([state, node_pool, edge_pool], dim=-1))
                )
        pooled = scatter_mean(nodes, batch.node_batch, batch.n_graphs)
        latent = self.readout(torch.cat([pooled, state], dim=-1))
        return self.head(latent).reshape(-1), latent


def build_native_model(name: str, input_dim: int | None, parameters: dict[str, Any]) -> nn.Module:
    common = {
        "hidden_dim": int(parameters.get("hidden_dim", 192)),
        "latent_dim": int(parameters.get("native_latent_dim", 256)),
        "layers": int(parameters.get("layers", 4)),
        "dropout": float(parameters.get("dropout", 0.08)),
    }
    if name == "native_modnet":
        if input_dim is None:
            raise ValueError("native_modnet requires input_dim")
        return NativeDescriptorMLP(input_dim, **common)
    architecture = name.removeprefix("native_")
    return NativeCrystalModel(
        architecture,
        **common,
        rbf_dim=int(parameters.get("rbf_dim", 64)),
        cutoff=float(parameters.get("cutoff", 8.0)),
        heads=int(parameters.get("heads", 6)),
    )
