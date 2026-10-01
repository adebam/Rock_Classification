import torch
import torchvision

from torch import nn

def create_ConvNeXt_Tiny_Weights_model(num_classes: int = 18, seed: int = 42):
    """Create an ImageNet-pretrained ConvNeXt-Tiny with a new classifier.

    Args:
        num_classes (int): Number of output classes. Defaults to 18.
        seed (int): Seed for the new classifier's initialization. Defaults to 42.

    Returns:
        model (torch.nn.Module): ConvNeXt-Tiny with all parameters trainable.
        transforms: Preprocessing transforms for the pretrained weights.

    Load your trained rock-classification weights and call model.eval()
    before using this model for inference. The new classifier is untrained.
    """
    weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.convnext_tiny(weights=weights)

    # Replace only the final linear layer, preserving normalization and flattening.
    torch.manual_seed(seed)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    return model, transforms
