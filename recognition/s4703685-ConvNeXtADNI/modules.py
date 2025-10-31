"""
 The term “ConvNeXt” is often used to refer to a type of deep learning model used for feature extraction on datasets in the learning process. 
 This type of model is part of a deep learning subfield called convolutional neural networks (CNN).
 ConvNext models consist of convolutional layers followed by fully connected layers.

 ConvNeXt model was created by following the resource in this GitHub repository: https://github.com/FrancescoSaverioZuppichini/ConvNext.
"""
from torch import nn
from torch import Tensor
from typing import List
import timm
import torch
from torchvision.ops import StochasticDepth
from torchvision.ops import StochasticDepth
"""
    A sequential container of convolutional, normalization, and activation layers.
    
    This utility layer composes (convolution) -> (normalization) -> (activation) operations
    in sequence for building neural network blocks.
    
    Args:
        in_features (int): Number of input channels/features
        out_features (int): Number of output channels/features
        kernel_size (int): Size of the convolutional kernel
        norm (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d
        act (nn.Module, optional): Activation function. Defaults to nn.GELU
        **kwargs: Additional arguments to pass to the convolutional layer
"""
class ConvNormAct(nn.Sequential):

    #A little util layer composed by (conv) -> (norm) -> (act) layers.

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm = nn.BatchNorm2d,
        act = nn.GELU,
        **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            norm(out_features),
            act(),
        )



"""
    Layer scaling module that applies learnable scaling factors to input tensors.
    
    This module introduces learnable scaling parameters (gamma) that are applied
    to the input tensor, helping with training stability and convergence.
    Reduces training time and improves generalization by randomly dropping entire 
    layers during training. Method allows for avoiding vanishing gradients and allowing for
    regularization.
    
    Args:
        init_value (float): Initial value for the scaling parameters
        dimensions (int): Number of dimensions/channels for the scaling parameters
"""
class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None,None] * x

"""
    Bottleneck block for ConvNeXt architecture.
    
    This block implements a bottleneck structure with depth-wise convolution,
    layer normalization, and stochastic depth for regularization. It follows
    a ResNet-like structure with transformer-inspired modifications.
    
    Args:
        in_features (int): Number of input channels
        out_features (int): Number of output channels
        expansion (int, optional): Expansion factor for intermediate features. Defaults to 4
        drop_p (float, optional): Drop path probability for stochastic depth. Defaults to 0.3
        layer_scaler_init_value (float, optional): Initial value for layer scaling. Defaults to 1e-6
"""
class BottleNeck(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = 0.3,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide 
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.shortcut = (
            nn.Conv2d(in_features, out_features, kernel_size=1)
            if in_features != out_features else nn.Identity()
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

        
    def forward(self, x: Tensor) -> Tensor:
        res = self.shortcut(x)
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x
    

"""
    A stage in the ConvNeXt architecture consisting of multiple bottleneck blocks.
    
    Each stage typically reduces the spatial dimensions by a factor of 2 and
    increases the channel dimensions. It contains a downsampling layer followed
    by multiple bottleneck blocks.
    
    Args:
        in_features (int): Number of input channels
        out_features (int): Number of output channels
        depth (int): Number of bottleneck blocks in this stage
        **kwargs: Additional arguments passed to BottleNeck blocks
"""
class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                BottleNeck(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )

"""
    Stem module for ConvNeXt that performs initial heavy downsampling.
    
    The stem is the first layer in the model that processes the input image
    and performs aggressive spatial reduction while expanding channels.
    
    Args:
        in_features (int): Number of input channels (typically 3 for RGB images)
        out_features (int): Number of output channels after the stem
"""
class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(
                in_features, out_features, kernel_size=4, stride=4
            ),
            nn.BatchNorm2d(out_features),
        )

"""
    ConvNeXt encoder that processes input through multiple stages.
    
    The encoder consists of a stem followed by a series of stages that
    progressively extract features at different spatial resolutions and
    channel dimensions.
    
    Args:
        in_channels (int): Number of input image channels
        stem_features (int): Number of output channels from the stem
        depths (List[int]): List of depths (number of blocks) for each stage
        widths (List[int]): List of widths (channel dimensions) for each stage
        drop_p (float, optional): Maximum drop path probability. Defaults to 0.3
"""
class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = 0.3,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))] 
        
        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )
        

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

"""
    Classification head for ConvNeXt that produces final class predictions.
    
    This head performs global average pooling, normalization, dropout,
    and final linear projection to the number of classes.
    
    Args:
        num_channels (int): Number of input channels from the encoder
        num_classes (int, optional): Number of output classes. Defaults to 1000
"""
class ClassificationHead(nn.Sequential):
    def __init__(self, num_channels: int, num_classes: int = 1000):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(num_channels),
            nn.Dropout(0.4), 
            nn.Linear(num_channels, num_classes)
        )
    
"""
    Complete ConvNeXt model for image classification tasks.
    
    This model combines the ConvNeXt encoder with a classification head
    to form a complete image classification pipeline.
    
    Args:
        in_channels (int): Number of input image channels
        stem_features (int): Number of output channels from the stem
        depths (List[int]): List of depths for each stage
        widths (List[int]): List of widths for each stage
        drop_p (float, optional): Maximum drop path probability. Defaults to 0.3
        num_classes (int, optional): Number of output classes. Defaults to 1000
"""    
class ConvNextForImageClassification(nn.Sequential):
    def __init__(self,  
                 in_channels: int,
                 stem_features: int,
                 depths: List[int],
                 widths: List[int],
                 drop_p: float = 0.3,
                 num_classes: int = 1000):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels, stem_features, depths, widths, drop_p)
        self.head = ClassificationHead(widths[-1], num_classes)

"""
    ADNI-specific ConvNeXt model for medical image classification.
    
    This is a specialized ConvNeXt model configured for ADNI dataset
    with specific architecture parameters optimized for the task.
    
    Args:
        in_features (int, optional): Number of input channels. Defaults to 3
        out_features (int, optional): Number of output classes. Defaults to 2
"""
class ADNIConvNeXt(nn.Module):
    def __init__(self, in_features=3, out_features=2):
        super().__init__()
        self.model = ConvNextForImageClassification(
            in_channels=in_features,
            stem_features=48,          # smaller model variant (ConvNeXt-Tiny)
            depths=[2,2,4,2],
            widths=[48, 96, 192, 384],
            drop_p=0.3,
            num_classes=out_features
        )

    def forward(self, x):
        # explicitly call encoder and head
        x = self.model.encoder(x)
        x = self.model.head(x)
        return x
    