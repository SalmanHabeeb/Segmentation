
import torch
from torch import nn, Tensor

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.downsample = downsample
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.GELU()
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList([self.conv1, self.conv2,])
        if self.downsample is not None:
            self.layers.append(self.downsample)
    
    def norm(self, x: Tensor) -> Tensor:
        if tuple(x.shape[-2:]) != (1, 1,):
            return  nn.InstanceNorm2d(self.out_channels)(x)
        else:
            return x

    def forward(self, x: Tensor) -> Tensor:
        result = self.layers[0](x)
        result = self.norm(result)
        result = self.activation(result)

        result = self.layers[1](result)
        result = self.norm(result)
        if self.downsample is not None:
            x = self.layers[2](x)
        
        result = result + x
        result = self.relu(result)

        return result

def get_basic_block(in_channels: int, out_channels: int, n_blocks: int=1, stride: int=1,) -> BasicBlock:
    downsample = None
    if (in_channels != out_channels) or (stride != 1):
        downsample = nn.Sequential(
                         nn.Conv2d(
                             in_channels, out_channels, kernel_size=3, stride=stride, padding=1
                         ),
                         nn.ReLU(inplace=True),
                     )
    
    blocks = [BasicBlock(in_channels, out_channels, stride, downsample=downsample)]
    for _ in range(n_blocks-1):
        blocks.append(BasicBlock(out_channels, out_channels, stride,))
    return nn.Sequential(*blocks)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, image_shape, n_layers, back_bone,):
        super(UNet, self).__init__()
        self.image_shape = image_shape
        self.back_bone = back_bone
        self.up = nn.ModuleList([get_basic_block(2048//4**i, 2048//4**(i+1), n_blocks=3) for i in range(n_layers)])
        self.out = nn.Sequential(
                       nn.Conv2d(2, out_channels, kernel_size=3, stride=1, padding=1),          
                       nn.Dropout(0.05),
                       nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                   )
    def forward(self, x: Tensor) -> Tensor:
        x = self.back_bone(x)
        x = x.reshape(*x.shape, 1, 1)
        for i, layer in enumerate(self.up):
            x = layer(x)
            if x.shape[-2:] != self.image_shape:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
        if x.shape[-2:] != self.image_shape:
            x = nn.UpsamplingBilinear2d(self.image_shape)(x)
        x = self.out(x)

        return x
