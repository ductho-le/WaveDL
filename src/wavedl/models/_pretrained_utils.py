"""
Shared Utilities for Model Architectures
=========================================

Common components used across multiple models:
- GRN (Global Response Normalization) for ConvNeXt V2
- Dimension-agnostic layer factories
- Regression head builders
- Input channel adaptation for pretrained models

Author: Ductho Le (ductho.le@outlook.com)
"""

import contextlib
import copy
import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DIMENSION-AGNOSTIC LAYER FACTORIES
# =============================================================================


def get_conv_layer(dim: int) -> type[nn.Module]:
    """Get dimension-appropriate Conv class."""
    layers = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    if dim not in layers:
        raise ValueError(f"Unsupported dimension: {dim}")
    return layers[dim]


def get_norm_layer(dim: int) -> type[nn.Module]:
    """Get dimension-appropriate BatchNorm class."""
    layers = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
    if dim not in layers:
        raise ValueError(f"Unsupported dimension: {dim}")
    return layers[dim]


def get_pool_layer(dim: int) -> type[nn.Module]:
    """Get dimension-appropriate AdaptiveAvgPool class."""
    layers = {1: nn.AdaptiveAvgPool1d, 2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}
    if dim not in layers:
        raise ValueError(f"Unsupported dimension: {dim}")
    return layers[dim]


# =============================================================================
# GLOBAL RESPONSE NORMALIZATION (GRN) - ConvNeXt V2
# =============================================================================


def _global_response_norm(x, gamma, beta, spatial_dims, eps):
    # Half-precision L2 norms can overflow even when every activation and the
    # final normalized response are representable. Keep the reduction and
    # modulation in float32, then cast back to x.dtype to preserve AMP dtype.
    is_low_prec = x.dtype in (torch.float16, torch.bfloat16)
    work = x.float() if is_low_prec else x
    gamma_w = gamma.to(dtype=work.dtype)
    beta_w = beta.to(dtype=work.dtype)
    response = torch.linalg.vector_norm(work, dim=spatial_dims, keepdim=True)
    response = response / (response.mean(dim=1, keepdim=True) + eps)
    return (gamma_w * (work * response) + beta_w + work).to(dtype=x.dtype)


class GRN1d(nn.Module):
    """
    Global Response Normalization for 1D inputs.

    GRN enhances inter-channel feature competition and promotes diversity.
    Replaces LayerScale in ConvNeXt V2.

    Reference: ConvNeXt V2 (CVPR 2023)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _global_response_norm(x, self.gamma, self.beta, (2,), self.eps)


class GRN2d(nn.Module):
    """
    Global Response Normalization for 2D inputs.

    GRN enhances inter-channel feature competition and promotes diversity.
    Replaces LayerScale in ConvNeXt V2.

    Reference: ConvNeXt V2 (CVPR 2023)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _global_response_norm(x, self.gamma, self.beta, (2, 3), self.eps)


class GRN3d(nn.Module):
    """
    Global Response Normalization for 3D inputs.

    GRN enhances inter-channel feature competition and promotes diversity.
    Replaces LayerScale in ConvNeXt V2.

    Reference: ConvNeXt V2 (CVPR 2023)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _global_response_norm(x, self.gamma, self.beta, (2, 3, 4), self.eps)


def get_grn_layer(dim: int) -> type[nn.Module]:
    """Get dimension-appropriate GRN class."""
    layers = {1: GRN1d, 2: GRN2d, 3: GRN3d}
    if dim not in layers:
        raise ValueError(f"Unsupported dimension: {dim}")
    return layers[dim]


# =============================================================================
# LAYER NORMALIZATION (Channels Last for CNNs)
# =============================================================================


class LayerNormNd(nn.Module):
    """
    LayerNorm that works with channels-first tensors of any dimension.
    Applies normalization over the channel dimension.
    """

    def __init__(self, normalized_shape: int, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim not in (1, 2, 3):
            raise ValueError(f"Unsupported dimension: {dim}. Expected 1, 2, or 3.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.dim = dim
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move channels to last, apply LN, move back
        if self.dim == 1:
            # (B, C, L) -> (B, L, C) -> LN -> (B, C, L)
            x = x.permute(0, 2, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 2, 1)
        elif self.dim == 2:
            # (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
        elif self.dim == 3:
            # (B, C, D, H, W) -> (B, D, H, W, C) -> LN -> (B, C, D, H, W)
            x = x.permute(0, 2, 3, 4, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")
        return x


# =============================================================================
# STOCHASTIC DEPTH (DropPath)
# =============================================================================


class DropPath(nn.Module):
    """
    Stochastic Depth (drop path) regularization for residual networks.

    Randomly drops entire residual branches during training. Used in modern
    architectures like ConvNeXt, Swin Transformer, UniRepLKNet.

    Args:
        drop_prob: Probability of dropping the path (default: 0.0)

    Reference:
        Huang, G., et al. (2016). Deep Networks with Stochastic Depth.
        https://arxiv.org/abs/1603.09382
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError("drop_prob must be between 0 and 1.")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        if self.drop_prob == 1.0:
            return x * 0.0

        keep_prob = 1.0 - self.drop_prob
        # Shape: (batch_size, 1, 1, ...) for broadcasting.
        # Sample and scale in FP32 so small drop probabilities (< 1/256) do not
        # quantize keep_prob to 1.0 or introduce gain drift under BF16/FP16 AMP.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (
            keep_prob + torch.rand(shape, dtype=torch.float32, device=x.device)
        ).floor_()
        return x * random_tensor.div_(keep_prob).to(dtype=x.dtype)


# =============================================================================
# BACKBONE FREEZING UTILITIES
# =============================================================================


def freeze_backbone(
    model: nn.Module,
    exclude_patterns: list[str] | None = None,
    *,
    eval_frozen: bool = True,
) -> int:
    """
    Freeze backbone parameters, keeping specified layers trainable.

    Note:
        ``BaseModel`` subclasses should prefer ``self._freeze_backbone_modules(...)``
        (with ``self.freeze_backbone = True``) so ``model.train()`` continues to keep
        frozen BatchNorm/Dropout submodules in ``eval()`` mode across training epochs.

    Args:
        model: The model whose parameters to freeze
        exclude_patterns: List of patterns to exclude from freezing.
            Module paths matching a pattern or its descendants stay trainable.
            Default: ["classifier", "head", "fc"]
        eval_frozen: Put frozen normalization/dropout submodules into eval mode
            so BatchNorm running stats do not drift during fine-tuning.

    Returns:
        Number of parameters frozen

    Example:
        >>> freeze_backbone(model.backbone, exclude_patterns=["fc", "classifier"])
    """
    if exclude_patterns is None:
        exclude_patterns = ["classifier", "head", "fc"]

    def _is_excluded(path: str) -> bool:
        return any(
            path == pattern or path.startswith(pattern + ".")
            for pattern in exclude_patterns
        )

    frozen_count = 0
    for name, param in model.named_parameters():
        if not _is_excluded(name):
            param.requires_grad = False
            frozen_count += param.numel()

    if eval_frozen:
        for mod_name, module in model.named_modules():
            if mod_name and not _is_excluded(mod_name):
                params = list(module.parameters())
                if not params or all(not p.requires_grad for p in params):
                    module.eval()

    return frozen_count


# =============================================================================
# REGRESSION HEAD BUILDERS
# =============================================================================


def build_regression_head(
    in_features: int,
    out_size: int,
    dropout_rate: float = 0.3,
    hidden_dim: int = 512,
) -> nn.Sequential:
    """
    Build a standard regression head for pretrained models.

    Args:
        in_features: Input feature dimension
        out_size: Number of regression targets
        dropout_rate: Dropout rate
        hidden_dim: Hidden layer dimension

    Returns:
        nn.Sequential regression head
    """
    mid_dim = max(hidden_dim // 2, 1)
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Dropout(dropout_rate * 0.5),
        nn.Linear(hidden_dim, mid_dim),
        nn.SiLU(inplace=True),
        nn.Linear(mid_dim, out_size),
    )


def adapt_input_channels(
    conv_layer: nn.Module,
    new_in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Adapt a convolutional layer for different input channels.

    For pretrained models, averages RGB weights to grayscale.

    Args:
        conv_layer: Original conv layer (1D/2D/3D)
        new_in_channels: New number of input channels (default: 1)
        pretrained: Whether to adapt pretrained weights

    Returns:
        New conv layer with adapted input channels
    """
    if not isinstance(conv_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise NotImplementedError(f"Unsupported layer type: {type(conv_layer)}")
    if (
        not isinstance(new_in_channels, Integral)
        or isinstance(new_in_channels, bool)
        or new_in_channels <= 0
    ):
        raise ValueError("new_in_channels must be a positive integer.")
    if conv_layer.groups != 1 and new_in_channels != conv_layer.in_channels:
        raise ValueError(
            "Changing input channels of a grouped convolution is unsupported."
        )

    # Preserve subclasses such as timm's Conv2dSame, whose dynamic padding is
    # implemented in forward(), as well as device, dtype and training mode.
    new_conv = copy.deepcopy(conv_layer)
    new_conv.in_channels = int(new_in_channels)
    new_conv.weight = nn.Parameter(
        conv_layer.weight.new_empty(
            conv_layer.out_channels,
            int(new_in_channels) // conv_layer.groups,
            *conv_layer.kernel_size,
        ),
        requires_grad=conv_layer.weight.requires_grad,
    )
    if pretrained:
        with torch.no_grad():
            if new_in_channels == conv_layer.in_channels:
                weight = conv_layer.weight
            elif new_in_channels == 1:
                weight = conv_layer.weight.mean(dim=1, keepdim=True)
            else:
                repeats = math.ceil(new_in_channels / conv_layer.in_channels)
                repeat_shape = (1, repeats) + (1,) * (conv_layer.weight.ndim - 2)
                weight = conv_layer.weight.repeat(*repeat_shape)[:, :new_in_channels]
                weight = weight * (conv_layer.in_channels / new_in_channels)
            new_conv.weight.copy_(weight)
            if conv_layer.bias is not None:
                new_conv.bias.copy_(conv_layer.bias)
    else:
        new_conv.reset_parameters()
    return new_conv


def adapt_first_conv_for_single_channel(
    module: nn.Module,
    conv_path: str,
    pretrained: bool = True,
) -> None:
    """
    Adapt the first convolutional layer of a pretrained model for single-channel input.

    This is a convenience function for torchvision-style models where the path
    to the first conv layer is known. It modifies the model in-place.

    For pretrained models, the RGB weights are averaged to create grayscale weights,
    which provides a reasonable initialization for single-channel inputs.

    Args:
        module: The model or submodule containing the conv layer
        conv_path: Dot-separated path to the conv layer (e.g., "conv1", "features.0.0")
        pretrained: Whether to adapt pretrained weights by averaging RGB channels

    Example:
        >>> # For torchvision ResNet
        >>> adapt_first_conv_for_single_channel(
        ...     model.backbone, "conv1", pretrained=True
        ... )
        >>> # For torchvision ConvNeXt
        >>> adapt_first_conv_for_single_channel(
        ...     model.backbone, "features.0.0", pretrained=True
        ... )
        >>> # For torchvision DenseNet
        >>> adapt_first_conv_for_single_channel(
        ...     model.backbone, "features.conv0", pretrained=True
        ... )
    """
    # Navigate to parent and get the conv layer
    parts = conv_path.split(".")
    parent = module
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    # Get the final attribute name and the old conv
    final_attr = parts[-1]
    if final_attr.isdigit():
        old_conv = parent[int(final_attr)]
    else:
        old_conv = getattr(parent, final_attr)

    # Create and set the new conv
    new_conv = adapt_input_channels(old_conv, new_in_channels=1, pretrained=pretrained)

    if final_attr.isdigit():
        parent[int(final_attr)] = new_conv
    else:
        setattr(parent, final_attr, new_conv)


def infer_backbone_features(
    backbone: nn.Module,
    in_shape: tuple[int, ...],
    in_channels: int = 1,
) -> int:
    """
    Infer the flattened feature dimension emitted by a backbone via a dummy pass.

    Several ``timm`` backbones (e.g. MIT EfficientViT) apply projection layers inside
    ``forward_head(..., pre_logits=True)`` when ``num_classes=0``, so ``backbone.num_features``
    differs from the actual output width. Probing in ``eval()`` mode preserves pretrained
    BatchNorm running statistics and supports ``batch_size=1`` on small spatial inputs.
    """
    was_training = backbone.training
    param_or_buf = next(backbone.parameters(), None)
    if param_or_buf is None:
        param_or_buf = next(backbone.buffers(), None)
    device = param_or_buf.device if param_or_buf is not None else torch.device("cpu")
    dtype = (
        param_or_buf.dtype
        if param_or_buf is not None and param_or_buf.is_floating_point()
        else torch.float32
    )
    backbone.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *in_shape, device=device, dtype=dtype)
            features = backbone(dummy)
            if features.ndim > 2:
                features = features.flatten(1)
            return int(features.shape[-1])
    finally:
        backbone.train(was_training)


def find_and_adapt_input_convs(
    backbone: nn.Module,
    pretrained: bool = True,
    adapt_all: bool = False,
) -> int:
    """
    Find and adapt Conv layers with 3 input channels for single-channel input.

    This is useful for timm-style models where the exact path to the first
    conv layer may vary or where multiple layers need adaptation.

    Args:
        backbone: The backbone model to adapt
        pretrained: Whether to adapt pretrained weights by averaging RGB channels
        adapt_all: If True, adapt all ungrouped 3-channel Conv layers.
                   If False (default), only adapt the first one found.

    Returns:
        Number of layers adapted

    Example:
        >>> # For timm models (adapt first conv only)
        >>> count = find_and_adapt_input_convs(model.backbone, pretrained=True)
        >>> # For models with multiple input convs (e.g., FastViT)
        >>> count = find_and_adapt_input_convs(
        ...     model.backbone, pretrained=True, adapt_all=True
        ... )
    """
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    adapted_count = 0
    adapted_ids: set[int] = set()

    for name, module in backbone.named_modules():
        if not name or id(module) in adapted_ids:
            continue
        if not hasattr(module, "in_channels") or module.in_channels != 3:
            continue

        # Check if this is a wrapper with inner .conv attribute
        if hasattr(module, "conv") and isinstance(module.conv, conv_types):
            old_conv = module.conv
            if old_conv.in_channels != 3 or old_conv.groups != 1:
                continue
            new_conv = adapt_input_channels(
                old_conv, new_in_channels=1, pretrained=pretrained
            )
            module.conv = new_conv
            with contextlib.suppress(AttributeError):
                module.in_channels = 1
            adapted_ids.add(id(old_conv))
            adapted_ids.add(id(new_conv))
            adapted_count += 1

        elif isinstance(module, conv_types):
            if module.groups != 1:
                continue
            # Direct Conv - need to replace it in parent
            parts = name.split(".")
            parent = backbone
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)

            child_name = parts[-1]
            new_conv = adapt_input_channels(
                module, new_in_channels=1, pretrained=pretrained
            )

            if child_name.isdigit():
                parent[int(child_name)] = new_conv
            else:
                setattr(parent, child_name, new_conv)

            adapted_ids.add(id(module))
            adapted_ids.add(id(new_conv))
            adapted_count += 1

        if not adapt_all and adapted_count > 0:
            break

    return adapted_count
