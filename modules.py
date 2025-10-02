"""
 The term “ConvNeXt” is often used to refer to a type of deep learning model used for feature extraction on datasets in the learning process. 
 This type of model is part of a deep learning subfield called convolutional neural networks (CNN).
 ConvNext models consist of convolutional layers followed by fully connected layers.
"""
from torch import nn
from torch import Tensor
from typing import List
import timm

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
        act = nn.ReLU,
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

class BottleNeck(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        reduction: int = 4,
        stride: int = 1,
    ):
        super().__init__()
        reduced_features = out_features // reduction
        self.block = nn.Sequential(
            # wide -> narrow
            ConvNormAct(
                in_features, reduced_features, kernel_size=1, stride=stride, bias=False
            ),
            # narrow -> narrow
            ConvNormAct(reduced_features, reduced_features, kernel_size=3, bias=False),
            # narrow -> wide
            ConvNormAct(reduced_features, out_features, kernel_size=1, bias=False, act=nn.Identity),
        )
        self.shortcut = (
            nn.Sequential(
                ConvNormAct(
                    in_features, out_features, kernel_size=1, stride=stride, bias=False
                )
            )
            if in_features != out_features
            else nn.Identity()
        )

        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        res = self.shortcut(res)
        x += res
        x = self.act(x)
        return x

#Check if the code above works    
import torch
x = torch.rand(1, 32, 7, 7)
block = BottleNeck(32, 64)
block(x).shape
print(block(x).shape)

#Code to define a stage, which is a sequence of bottleneck blocks that reduce in size as stages progress by a factor of 2.
class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, stride: int = 2, **kwargs
    ):
        super().__init__(
            # downsample is done here
            BottleNeck(in_features, out_features, stride=stride, **kwargs),
            *[
                BottleNeck(out_features, out_features, **kwargs)
                for _ in range(depth - 1)
            ],
        )

#Check staging class works as intended
stage = ConvNexStage(32, 64, depth=2)
print(stage(x).shape)


