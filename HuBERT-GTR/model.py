"""Optional HuBERT classifiers; not the SVMs reported in the 2024 paper."""
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, inputs):
        # A final singleton batch must not crash or update statistics from one item.
        if self.training and inputs.shape[0] == 1:
            return F.batch_norm(inputs, self.running_mean, self.running_var,
                                self.weight, self.bias, False, self.momentum, self.eps)
        return super().forward(inputs)


class _GTRBase(nn.Module):
    def __init__(self, model_path, output_features, *, backbone=None,
                 freeze_backbone=True, local_files_only=True, head_name="classifier"):
        super().__init__()
        if backbone is None:
            if not model_path:
                raise ValueError("Specify a local or explicitly selected Hugging Face model")
            from transformers import HubertModel
            backbone = HubertModel.from_pretrained(model_path, local_files_only=local_files_only)
        self.hubert_model = backbone
        self.freeze_backbone = freeze_backbone
        self.hubert_model.requires_grad_(not freeze_backbone)
        hidden = self.hubert_model.config.hidden_size
        self.fc1 = nn.Linear(hidden, 512)
        self.bn1 = SafeBatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_features)
        self.add_module(head_name, nn.Sequential(self.fc1, self.bn1, nn.ReLU(), self.dropout, self.fc2))
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.hubert_model.eval()
        return self

    def set_training(self, flag):
        self.train(flag)

    def _encode(self, inputs, attention_mask=None):
        if inputs.ndim == 3 and inputs.shape[1] == 1:
            inputs = inputs[:, 0]
        if inputs.ndim != 2:
            raise ValueError("Expected input_values shaped [batch,samples]")
        if attention_mask is not None and attention_mask.shape != inputs.shape:
            raise ValueError("attention_mask must match unchannelled input_values")
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            hidden = self.hubert_model(inputs, attention_mask=attention_mask).last_hidden_state
        if attention_mask is None:
            return hidden.mean(dim=1)
        if hasattr(self.hubert_model, "_get_feature_vector_attention_mask"):
            mask = self.hubert_model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        else:
            lengths = attention_mask.sum(dim=-1)
            for kernel, stride in zip(self.hubert_model.config.conv_kernel, self.hubert_model.config.conv_stride):
                lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
            if (lengths < 1).any():
                raise ValueError("An input is shorter than the encoder receptive field")
            mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class GTRClassifier(_GTRBase):
    def __init__(self, model_path=None, num_classes=5, **kwargs):
        super().__init__(model_path, num_classes, head_name="classifier", **kwargs)

    def forward(self, inputs, attention_mask=None):
        return self.classifier(self._encode(inputs, attention_mask))


class GTRRegressor(_GTRBase):
    def __init__(self, model_path=None, **kwargs):
        super().__init__(model_path, 1, head_name="regressor", **kwargs)

    def forward(self, inputs, attention_mask=None):
        return self.regressor(self._encode(inputs, attention_mask))
