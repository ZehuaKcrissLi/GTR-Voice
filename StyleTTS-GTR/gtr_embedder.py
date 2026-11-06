"""GTR label embedders, with explicit legacy and paper-described variants.

The paper does not publish joint-label encoding or layer widths.  Consequently
the single-embedding variant below is a documented derived configuration, not
an assertion that its parameters match the missing paper checkpoint.
"""
import torch
from torch import nn


def validate_labels(labels):
    if labels.ndim != 1 or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("GTR labels must be a one-dimensional integer tensor")
    g, t, r = labels // 100, (labels // 10) % 10, labels % 10
    valid = (g >= 0) & (g <= 4) & (t >= 1) & (t <= 5) & (r >= 0) & (r <= 6)
    valid &= (g == 0) == (r == 0)
    if not bool(valid.all()):
        raise ValueError("Illegal GTR label: G=0/R=0 must be coupled; T is 1..5")
    return g.long(), t.long(), r.long()


class GTRStyleEncoder(nn.Module):
    """Released three-embedding architecture; default state-dict keys preserved."""
    def __init__(self, out_dim=128, style_dim=4):
        super().__init__()
        self.g_embedding = nn.Embedding(5, style_dim)
        self.t_embedding = nn.Embedding(6, style_dim)
        self.r_embedding = nn.Embedding(7, style_dim)
        self.fc_sequnetial = nn.Sequential(
            nn.Linear(style_dim * 3, 48), nn.ReLU(),
            nn.Linear(48, 96), nn.ReLU(), nn.Linear(96, out_dim))

    def forward(self, labels):
        g, t, r = validate_labels(labels)
        return self.fc_sequnetial(torch.cat(
            [self.g_embedding(g), self.t_embedding(t), self.r_embedding(r)], dim=1))


class PaperDescribedGTRStyleEncoder(nn.Module):
    """One joint embedding plus two FC layers; widths/encoding are derived."""
    def __init__(self, out_dim=128, style_dim=32, hidden_dim=64):
        super().__init__()
        combinations = [(g, t, r) for g in range(5) for t in range(1, 6)
                        for r in range(7) if (g == 0) == (r == 0)]
        lookup = torch.full((457,), -1, dtype=torch.long)
        for index, (g, t, r) in enumerate(combinations):
            lookup[100 * g + 10 * t + r] = index
        self.register_buffer("label_to_index", lookup, persistent=False)
        self.embedding = nn.Embedding(125, style_dim)
        self.fc = nn.Sequential(nn.Linear(style_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, out_dim))

    def forward(self, labels):
        validate_labels(labels)
        return self.fc(self.embedding(self.label_to_index[labels.long()]))


def make_embedder(config):
    options = dict(config)
    architecture = options.pop("architecture")
    if architecture == "legacy":
        return GTRStyleEncoder(**options)
    if architecture == "paper-described":
        return PaperDescribedGTRStyleEncoder(**options)
    raise ValueError("architecture must be legacy or paper-described")
