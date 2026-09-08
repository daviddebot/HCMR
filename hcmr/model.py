"""Core H-CMR model.

The model learns (1) a directed acyclic concept graph from node priorities,
(2) a memory of three-valued logic rules, and (3) an attention distribution
over rules for every target node.  Root nodes are predicted by an image encoder;
non-root nodes are inferred by applying selected rules hierarchically.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _straight_through_binary(probability: Tensor) -> Tensor:
    hard = (probability > 0.5).to(probability.dtype)
    return probability + (hard - probability).detach()


class DigitPairEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 4 * 4, embedding_dim // 2), nn.ReLU(),
        )
        self.project = nn.Sequential(nn.Linear(embedding_dim // 2, embedding_dim // 2), nn.ReLU())

    def forward(self, images: Tensor) -> Tensor:
        left = self.project(self.cnn(images[:, 0]))
        right = self.project(self.cnn(images[:, 1]))
        return torch.cat((left, right), dim=-1)


class HCMR(nn.Module):
    """Hierarchical Concept Memory Reasoner.

    Args:
        n_nodes: Number of concept and task nodes.
        n_rules: Number of logic rules in the memory.
        embedding_dim: Image-encoder representation size.
        graph_temperature: Temperature used to relax graph orientation.
    """

    def __init__(self, n_nodes: int, n_rules: int = 10, embedding_dim: int = 128,
                 graph_temperature: float = 1.0):
        super().__init__()
        self.n_nodes, self.n_rules = n_nodes, n_rules
        self.graph_temperature = graph_temperature
        self.encoder = DigitPairEncoder(embedding_dim)
        self.concept_head = nn.Linear(embedding_dim, n_nodes)
        self.concept_embedding = nn.Parameter(torch.randn(n_nodes, embedding_dim // 4) * 0.02)
        self.node_priority = nn.Parameter(torch.zeros(n_nodes))
        # Rule states are positive, negative, and irrelevant respectively.
        self.rule_logits = nn.Parameter(torch.zeros(n_rules, n_nodes, n_nodes, 3))
        self.selector = nn.Sequential(
            nn.Linear(n_nodes + embedding_dim // 4, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, n_nodes * n_rules),
        )

    def orientation(self, hard: bool | None = None) -> Tensor:
        """Return source-to-target DAG orientation induced by node priorities."""
        difference = self.node_priority[None, :] - self.node_priority[:, None]
        soft = torch.sigmoid(difference / self.graph_temperature)
        soft = soft * (1 - torch.eye(self.n_nodes, device=soft.device))
        if hard is None:
            hard = self.training
        return _straight_through_binary(soft) if hard else (soft > 0.5).float()

    def rule_probabilities(self) -> Tensor:
        rules = self.rule_logits.softmax(dim=-1)
        diagonal = torch.eye(self.n_nodes, device=rules.device, dtype=rules.dtype)
        diagonal = diagonal.unsqueeze(0).unsqueeze(-1)
        self_role = torch.tensor((0.0, 0.0, 1.0), device=rules.device, dtype=rules.dtype)
        return rules * (1 - diagonal) + self_role * diagonal

    def _rule_values(self, state: Tensor, rules: Tensor) -> Tensor:
        # state: [batch, source], rules: [rule, source, target, role]
        positive = state[:, None, :, None]
        negative = 1 - positive
        irrelevant = torch.ones_like(positive)
        truth = (rules[..., 0] * positive + rules[..., 1] * negative + rules[..., 2] * irrelevant)
        return truth.prod(dim=2).permute(0, 2, 1)  # [batch, target, rule]

    def forward(self, images: Tensor, interventions: Tensor | None = None,
                intervention_mask: Tensor | None = None, unroll_steps: int | None = None):
        embedding = self.encoder(images)
        base_prob = torch.sigmoid(self.concept_head(embedding))
        orientation = self.orientation()
        rules = self.rule_probabilities()
        # A rule contributes only across an oriented graph edge.
        signed_roles = rules[..., :2] * orientation[None, :, :, None]
        irrelevant_role = 1 - signed_roles.sum(dim=-1, keepdim=True)
        roles = torch.cat((signed_roles, irrelevant_role), dim=-1)
        parents = (1 - roles[..., 2].prod(dim=0)).clamp(0, 1)
        roots = parents.sum(dim=0).eq(0)

        state = base_prob
        steps = 1 if self.training else (unroll_steps or self.n_nodes)
        for _ in range(steps):
            pooled_embedding = parents.T @ self.concept_embedding
            selector_input = torch.cat((state, pooled_embedding.unsqueeze(0).expand(len(state), -1, -1).mean(dim=1)), dim=1)
            attention = self.selector(selector_input).view(-1, self.n_nodes, self.n_rules).softmax(dim=-1)
            rule_value = self._rule_values(state, roles)
            inferred = (attention * rule_value).sum(dim=-1)
            next_state = torch.where(roots.unsqueeze(0), base_prob, inferred)
            if interventions is not None and intervention_mask is not None:
                next_state = torch.where(intervention_mask.bool(), interventions, next_state)
            if not self.training and torch.equal(next_state.detach() >= .5, state.detach() >= .5):
                state = next_state
                break
            state = next_state
        return {"base_prob": base_prob, "node_prob": state, "orientation": orientation,
                "parents": parents, "rules": rules, "attention": attention}

