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

#First implement a resnet model as ConvNeXt starts with a ResNet architecture and then applies a series of transformer-inspired design changes to it.
class ConvNormAct(nn.Sequential):
    """
    A little util layer composed by (conv) -> (norm) -> (act) layers.
    """
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

#Reduces training time and improves generalization by randomly dropping entire layers during training. Method allows for avoiding vanishing gradients and allowing for
#regularization.
#VAnishing gradient - a problem in deep neural network training where the gradients become extremely small as they are backpropagated from the output to the earlier layers.
#regularization - techniques used to prevent overfitting in machine learning models by adding constraints or penalties to the model's complexity.
from torchvision.ops import StochasticDepth

class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None,None] * x

class BottleNeck(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = .0,
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
#Check if the code above works    
x = torch.rand(1, 32, 7, 7)
block = BottleNeck(32, 64)
"""

#Code to define a stage, which is a sequence of bottleneck blocks that reduce in size as stages progress by a factor of 2.
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

#Check staging class works as intended
"""
stage = ConvNexStage(32, 64, depth=1)
stage(torch.randn(1, 32, 14, 14)).shape
"""
#Simulates the first layer in the model that does the heavy downsampling of the input image.
class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(
                in_features, out_features, kernel_size=4, stride=4
            ),
            nn.BatchNorm2d(out_features),
        )

#A class that holds a list of stages and takes an image as input producing the final embeddings.
class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = 0.0,
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
#code to check if the ConvNextEncoder works as intended
image = torch.rand(1, 3, 224, 224)
encoder = ConvNextEncoder(in_channels=3, stem_features=64, depths=[3,4,6,4], widths=[256, 512, 1024, 2048])
print(encoder(image).shape)
print(stage(torch.randn(1, 32, 14, 14)).shape)
"""
class ClassificationHead(nn.Sequential):
    def __init__(self, num_channels: int, num_classes: int = 1000):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_classes)
        )
    
    
class ConvNextForImageClassification(nn.Sequential):
    def __init__(self,  
                 in_channels: int,
                 stem_features: int,
                 depths: List[int],
                 widths: List[int],
                 drop_p: float = .0,
                 num_classes: int = 1000):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels, stem_features, depths, widths, drop_p)
        self.head = ClassificationHead(widths[-1], num_classes)


