"""
gnn_model.py — GraphTransformerValidator.

A graph transformer (stacked torch_geometric.nn.TransformerConv, the PyG
implementation of Shi et al. 2021's "Masked Label Prediction" / UniMP-style
graph attention transformer) rather than poc_validator.py's plain 2-layer
GCN: TransformerConv does multi-head attention over each node's neighborhood
*conditioned on edge_attr*, so relation type/confidence/support_count
directly shape attention weights instead of only entering the graph as
uniform-weight connectivity. That distinction matters for exactly what this
validator has to detect — a hallucinated_relations edge and a real one look
identical in plain topology, so the signal is almost entirely in the edge
features TransformerConv can attend over and a bare GCNConv can't.

Configurable width/depth/heads, kept as constructor args (matching the
existing GCN/GAT-configurable pattern) so a hyperparameter sweep doesn't
need code changes.

Element heads score every node and edge ("part of a defect?") from the same
embeddings the graph readout pools. `element_heads=False` builds the original
graph-only model, for checkpoints trained before they existed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_max_pool, global_mean_pool

from kg_dataset.corruption import HEAD_NAMES_EXTENDED


class GraphTransformerValidator(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        head_names: list[str] | None = None,
        element_heads: bool = True,
    ):
        super().__init__()
        self.head_names = head_names or list(HEAD_NAMES_EXTENDED)
        self.dropout = dropout
        self.element_heads = element_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        head_dim = hidden_dim // num_heads

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    edge_dim=edge_feature_dim,
                    dropout=dropout,
                    beta=True,  # gated residual inside the conv itself (Shi et al.)
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Graph-level embedding = mean-pool || max-pool, same idea as
        # concatenating multiple readouts in Xu et al. 2018 (GIN) — mean
        # captures overall composition, max captures the presence of any
        # strongly anomalous node (e.g. a single orphan-injected entity),
        # which a graph-level defect can hinge on even in a large graph.
        readout_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.head_names)),
        )

        if element_heads:
            self.node_head = nn.Sequential(
                nn.Linear(hidden_dim + readout_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.edge_head = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h + residual)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward_all(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Raw logits: "graph" [num_graphs, num_heads], plus "node" [num_nodes] and
        "edge" [num_edges] with element heads (both edge directions are scored)."""
        h = self._encode(x, edge_index, edge_attr)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        out = {"graph": self.classifier(pooled)}

        if self.element_heads:
            out["node"] = self.node_head(torch.cat([h, pooled[batch]], dim=-1)).squeeze(-1)
            src, tgt = edge_index
            out["edge"] = self.edge_head(torch.cat([h[src], h[tgt], edge_attr], dim=-1)).squeeze(-1)
        return out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = torch.sigmoid(self.forward_logits(x, edge_index, edge_attr, batch))
        return {name: scores[:, i] for i, name in enumerate(self.head_names)}

    def forward_logits(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Raw [num_graphs, num_heads] logits — used by train.py so the loss
        can go through BCEWithLogitsLoss instead of BCE-on-sigmoid-output,
        which is numerically safer at the extremes."""
        h = self._encode(x, edge_index, edge_attr)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        return self.classifier(pooled)
