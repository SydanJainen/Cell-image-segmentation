import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchviz import make_dot
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class double_convolution(nn.Module):
    """
    This class implements the double convolution block which consists of two 3X3 convolution layers,
    each followed by a ReLU activation function.

    """
    def __init__(self, in_channels, out_channels): # Initialize the class
        super().__init__() # Initialize the parent class

        # First 3X3 convolution layer
        self.first_cnn = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1)
        self.act1 = nn.ReLU()

        # Second 3X3 convolution layer
        self.second_cnn = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1)
        self.act2 = nn.ReLU()

    # Pass the input through the double convolution block
    def forward(self, x):
        x = self.first_cnn(x)
        x = self.act1(x)
        x = self.act2(self.second_cnn(x))
        return x
    
class triple_convolution(nn.Module):
    """
    This class implements the triple convolution block which consists of three 3X3 convolution layers,
    each followed by a ReLU activation function.

    """
    def __init__(self, in_channels, out_channels): # Initialize the class
        super().__init__() # Initialize the parent class

        # First 3X3 convolution layer
        self.first_cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        # Second 3X3 convolution layer
        self.second_cnn = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        # Third 3X3 convolution layer
        self.third_cnn = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

    # Pass the input through the triple convolution block
    def forward(self, x):
        x = self.first_cnn(x)
        x = self.act1(x)
        x = self.second_cnn(x)
        x = self.act2(x)
        x = self.third_cnn(x)
        x = self.act3(x)
        return x

# Implement the Downsample block that occurs after each double convolution block
class down_sample(nn.Module):
    """
    This class implements the downsample block which consists of a Max Pooling layer with a kernel size of 2.
    The Max Pooling layer halves the image size reducing the spatial resolution of the feature maps
    while retaining the most important features.
    """
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    # Pass the input through the downsample block
    def forward(self, x):
        x = self.max_pool(x)
        return x
    
# Implement the UpSample block that occurs in the decoder part of the network
class up_sample(nn.Module):
    """
    This class implements the upsample block which consists of a convolution transpose layer with a kernel size of 2.
    The convolution transpose layer doubles the image size increasing the spatial resolution of the feature maps
    while retaining the learned features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convolution transpose layer
        self.up_sample = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, stride = 2)

    # Pass the input through the upsample block
    def forward(self, x):
        x = self.up_sample(x)
        return x 

# Implement the crop and concatenate block that occurs in the decoder part of the network
# This block concatenates the output of the upsample block with the output of the corresponding downsample block
# The output of the crop and concatenate block is then passed through a double convolution block
class crop_and_concatenate(nn.Module):
    """
    This class implements the crop and concatenate block which combines the output of the upsample block
    with the corresponding features from the contracting path through skip connections,
    allowing the network to recover the fine-grained details lost during downsampling
    and produce a high-resolution output segmentation map.
    """ 
    # def forward(self, upsampled, bypass):
    #     # Crop the feature map from the contacting path to match the size of the upsampled feature map
    #     bypass = torchvision.transforms.functional.center_crop(img = bypass, output_size = [upsampled.shape[2], upsampled.shape[3]]) 
    #     # Concatenate the upsampled feature map with the cropped feature map from the contracting path
    #     x = torch.cat([upsampled, bypass], dim = 1) # Concatenate along the channel dimension
    #     return x
    # Alternatively crop the upsampled feature map to match the size of the feature map from the contracting path
    def forward(self, upsampled, bypass):
        upsampled = torchvision.transforms.functional.resize(img = upsampled, size = bypass.shape[2:], antialias=True)
        x = torch.cat([upsampled, bypass], dim = 1) # Concatenate along the channel dimension
        return x

# m = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
# input = torch.randn(1, 1024, 28, 28)
# m(input).shape 

# m = nn.MaxPool2d(kernel_size = 2, stride = 2)
# xx = torch.randn(1, 1, 143, 143)
# m(xx).shape

## Implement the UNet architecture
class UNet(nn.Module):
    # in_channels: number of channels in the input image
    # out_channels: number of channels in the output image
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Define the contracting path: convolution blocks followed by downsample blocks
        self.down_conv = nn.ModuleList(double_convolution(in_chans, out_chans) for in_chans, out_chans in
                                       [(in_channels, 64), (64, 128), (128, 256), (256, 512)]) # List of downsample blocks
        
        self.down_samples = nn.ModuleList(down_sample() for _ in range(4))

        # Define the bottleneck layer
        self.bottleneck = double_convolution(in_channels = 512, out_channels = 1024)

        # Define the expanding path: upsample blocks followed by convolution blocks
        self.up_samples = nn.ModuleList(up_sample(in_chans, out_chans) for in_chans, out_chans in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]) # List of upsample blocks
        
        self.concat = nn.ModuleList(crop_and_concatenate() for _ in range(4))

        self.up_conv = nn.ModuleList(double_convolution(in_chans, out_chans) for in_chans, out_chans in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]) # List of convolution blocks
        
        # Final 1X1 convolution layer to produce the output segmentation map:
        # The primary purpose of 1x1 convolutions is to transform the channel dimension of the feature map,
        # while leaving the spatial dimensions unchanged.
        self.final_conv = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size = 1)

    # Pass the input through the UNet architecture
    def forward(self, x):
        # Pass the input through the contacting path
        skip_connections = [] # List to store the outputs of the downsample blocks
        for down_conv, down_sample in zip(self.down_conv, self.down_samples):
            x = down_conv(x)
            skip_connections.append(x)
            x = down_sample(x)
        
        # Pass the output of the contacting path through the bottleneck layer
        x = self.bottleneck(x)

        # Pass the output of the bottleneck layer through the expanding path
        skip_connections = skip_connections[::-1] # Reverse the list of skip connections
        for up_sample, concat, up_conv in zip(self.up_samples, self.concat, self.up_conv):
            x = up_sample(x)
            x = concat(x, skip_connections.pop(0)) # Remove the first element from the list of skip connections
            x = up_conv(x)
        
        # Pass the output of the expanding path through the final convolution layer
        x = self.final_conv(x)
        return x


def visualize_model_layers(model):
    for i, layer in enumerate(model.children()):
        print(f'Layer {i}: {layer}')

model = UNet(in_channels = 3, out_channels = 1).to(device)
dummy_input = torch.randn((1, 3, 32, 32)).to(device)
mask = model(dummy_input)
print(mask.shape)
visualize_model_layers(model)

