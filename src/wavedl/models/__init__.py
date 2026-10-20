"""
Model Registry and Factory Pattern for Deep Learning Architectures
===================================================================

This module provides a centralized registry for neural network architectures,
enabling dynamic model selection via command-line arguments.

**Dimensionality Coverage**:
    - 1D (waveforms): TCN, CNN, ResNet, ConvNeXt, ConvNeXt V2, DenseNet, ViT, Mamba, WaveNet, S4D
    - 2D (images): CNN, ResNet, ConvNeXt, ConvNeXt V2, DenseNet, ViT, UNet,
                   EfficientNet, MobileNetV3, RegNet, Swin, MaxViT, FastViT,
                   CAFormer, PoolFormer, Vision Mamba, EfficientViT
    - 3D (volumes): ResNet3D, CNN, ResNet, ConvNeXt, ConvNeXt V2, DenseNet

**Output Coverage**:
    - Vector regression: every registered architecture except ``unet_field``
    - Dense 2D regression: ``unet_field``

Usage:
    from wavedl.models import get_model, list_models, MODEL_REGISTRY

    # List available models
    print(list_models())

    # Get a model class by name
    ModelClass = get_model("cnn")
    model = ModelClass(in_shape=(500, 500), out_size=5)

Adding New Models:
    1. Create a new file in models/ (e.g., models/my_model.py)
    2. Inherit from BaseModel
    3. Use the @register_model decorator

    Example:
        from wavedl.models.base import BaseModel
        from wavedl.models.registry import register_model

        @register_model("my_model")
        class MyModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                ...

Author: Ductho Le (ductho.le@outlook.com)
"""

# Import registry first (no dependencies)
# Import base class (depends only on torch)
from .base import BaseModel

# Import model implementations (triggers registration via decorators)
from .cnn import CNN
from .convnext import (
    ConvNeXtBase_,
    ConvNeXtSmall,
    ConvNeXtTiny,
    ConvNeXtTinyPretrained,
)

# New models (v1.6+)
from .convnext_v2 import (
    ConvNeXtV2Base,
    ConvNeXtV2BaseLarge,
    ConvNeXtV2Small,
    ConvNeXtV2Tiny,
    ConvNeXtV2TinyPretrained,
)
from .densenet import DenseNet121, DenseNet121Pretrained, DenseNet169
from .efficientnet import EfficientNetB0, EfficientNetB2, EfficientNetB4, EfficientNetB7
from .efficientnetv2 import EfficientNetV2L, EfficientNetV2M, EfficientNetV2S
from .mamba import Mamba1D, VimBase, VimSmall, VimTiny
from .mobilenetv3 import MobileNetV3Large, MobileNetV3Small
from .registry import (
    MODEL_REGISTRY,
    build_model,
    get_model,
    import_model_modules,
    list_models,
    register_model,
)
from .regnet import RegNetY1_6GF, RegNetY3_2GF, RegNetY8GF, RegNetY400MF, RegNetY800MF
from .resnet import (
    ResNet18,
    ResNet18Pretrained,
    ResNet34,
    ResNet50,
    ResNet50Pretrained,
)
from .resnet3d import MC3_18, ResNet3D18
from .s4d import S4D, S4DLarge, S4DSmall
from .swin import SwinBase, SwinSmall, SwinTiny
from .tcn import TCN, TCNLarge, TCNSmall
from .unet import UNetField, UNetRegression
from .vit import ViTBase_, ViTSmall, ViTTiny
from .wavenet import WaveNet, WaveNetLarge, WaveNetSmall


# Optional timm-based models (imported conditionally)
try:
    from .caformer import CaFormerM36, CaFormerS18, CaFormerS36, PoolFormerS12
    from .efficientvit import (
        EfficientViTB1,
        EfficientViTB2,
        EfficientViTL2,
        EfficientViTM1,
    )
    from .fastvit import FastViTS12, FastViTSA12, FastViTT8, FastViTT12
    from .maxvit import MaxViTBaseLarge, MaxViTSmall, MaxViTTiny
    from .unireplknet import (
        UniRepLKNetBaseLarge,
        UniRepLKNetSmall,
        UniRepLKNetTiny,
    )

    _HAS_TIMM_MODELS = True
except ImportError:
    _HAS_TIMM_MODELS = False


# Export public API (sorted alphabetically per RUF022)
# See module docstring for dimensionality support details
__all__ = [
    "CNN",
    "MC3_18",
    "MODEL_REGISTRY",
    "S4D",
    "TCN",
    # Classes (uppercase first, alphabetically)
    "BaseModel",
    "ConvNeXtBase_",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
    "ConvNeXtTinyPretrained",
    "ConvNeXtV2Base",
    "ConvNeXtV2BaseLarge",
    "ConvNeXtV2Small",
    "ConvNeXtV2Tiny",
    "ConvNeXtV2TinyPretrained",
    "DenseNet121",
    "DenseNet121Pretrained",
    "DenseNet169",
    "EfficientNetB0",
    "EfficientNetB2",
    "EfficientNetB4",
    "EfficientNetB7",
    "EfficientNetV2L",
    "EfficientNetV2M",
    "EfficientNetV2S",
    "Mamba1D",
    "MobileNetV3Large",
    "MobileNetV3Small",
    "RegNetY1_6GF",
    "RegNetY3_2GF",
    "RegNetY8GF",
    "RegNetY400MF",
    "RegNetY800MF",
    "ResNet3D18",
    "ResNet18",
    "ResNet18Pretrained",
    "ResNet34",
    "ResNet50",
    "ResNet50Pretrained",
    "S4DLarge",
    "S4DSmall",
    "SwinBase",
    "SwinSmall",
    "SwinTiny",
    "TCNLarge",
    "TCNSmall",
    "UNetField",
    "UNetRegression",
    "ViTBase_",
    "ViTSmall",
    "ViTTiny",
    "VimBase",
    "VimSmall",
    "VimTiny",
    "WaveNet",
    "WaveNetLarge",
    "WaveNetSmall",
    # Functions (lowercase, alphabetically)
    "build_model",
    "get_model",
    "import_model_modules",
    "list_models",
    "register_model",
]

# Add timm-based models to __all__ if available
if _HAS_TIMM_MODELS:
    __all__.extend(
        [
            "CaFormerM36",
            "CaFormerS18",
            "CaFormerS36",
            "EfficientViTB1",
            "EfficientViTB2",
            "EfficientViTL2",
            "EfficientViTM1",
            "FastViTS12",
            "FastViTSA12",
            "FastViTT8",
            "FastViTT12",
            "MaxViTBaseLarge",
            "MaxViTSmall",
            "MaxViTTiny",
            "PoolFormerS12",
            "UniRepLKNetBaseLarge",
            "UniRepLKNetSmall",
            "UniRepLKNetTiny",
        ]
    )
