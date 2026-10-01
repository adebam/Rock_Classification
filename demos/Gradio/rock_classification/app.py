
#####################################
#        Needs
######################################
from torch.utils.data import DataLoader
from pathlib import Path
from functools import partial
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights


APP_DIR = Path(__file__).resolve().parent

def remove_bottom_annotation(image, pixels):
    width, height = image.size

    if not 0 <= pixels < height:
        raise ValueError("Image height must exceed the crop size.")

    return image.crop((0, 0, width, height - pixels))


train_eval_transform = transforms.Compose([
    transforms.Lambda(
        partial(remove_bottom_annotation, pixels=90)
    ),
    ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms(),
])

#####################################
#Gradio
######################################
### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_ConvNeXt_Tiny_Weights_model
from timeit import default_timer as timer
from typing import Tuple, Dict

### 2. Model and transforms preparation ###
# convNext model
ConvNeXt_Tiny, ConvNeXt_Tiny_transforms = create_ConvNeXt_Tiny_Weights_model(num_classes=18,seed=42)

ConvNeXt_Tiny.load_state_dict(
    torch.load(APP_DIR/"INFERENCE_MERGED__18_class_convnext_tiny_reproduce_adamw_trial21_epoch11.pth", map_location="cpu", weights_only=True)
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = train_eval_transform(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    ConvNeXt_Tiny.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(ConvNeXt_Tiny(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time
#class list
class_names = [
    "class2_class13", "class3", "class4", "class5",
    "class6", "class7", "class8", "class9",
    "class10", "class11", "class12", "class14",
    "class15", "class16", "class17", "class18_class19",
    "class20", "class21_class22",
]

### 4. Gradio app ###

# Create title, description and article strings
title = "Rock Classification 🪨"
description = "A ConvNeXt_Tiny feature extractor computer vision model to classify images of rocks."
article = "Created at [05_Gradio_on_Huggingface.ipynb](https://github.com/adebam/Rock_Classification)."

# Create examples list from "examples/" directory 
#APP_DIR = Path(__file__).resolve().parent
examples_dir = APP_DIR / "examples"
example_list = [[str(path)] for path in sorted(examples_dir.glob("*.jpg"))]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
