import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    TinyVGG-style convolutional neural network for image classification.
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self,input_shape: int,hidden_units: int,output_shape: int,pool_output_size: int = 4,dropout: float = 0.0,padding: int = 0):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3, # how big is the square that's going over the image?
                stride=1, # default
                padding=padding), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=padding),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            # Converts any H × W feature map into 1 × 1
            nn.AdaptiveAvgPool2d((pool_output_size, pool_output_size)), # can experiment with this later
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_units * pool_output_size * pool_output_size,# i dont need the trick anymore. I am using adaptive pooling
                out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x))
        )



class DINOv2Classifier(nn.Module):
    def __init__(
        self,
        output_shape: int,
        freeze_backbone: bool = False
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(
            self.backbone.embed_dim,
            output_shape
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)