import streamlit as st
import os
import torch
import torch.nn as nn
import pandas as pd
import json
import boto3
from torchvision import models, transforms
from functools import partial
from torchvision.models import ConvNeXt_Tiny_Weights
# ============================================================
# CONFIGURATION
# ============================================================
BUCKET_NAME = "super-dayo-409400548716"

S3_PREFIX = "models/rock-classifier/convnext_tiny_18class/"

MODEL_FILENAME = (
    "INFERENCE_MERGED__18_class_convnext_tiny_"
    "reproduce_adamw_trial21_epoch11.pth"
)

S3_MODEL_KEY = S3_PREFIX + MODEL_FILENAME

local_directory="S3data_download"


CONFIDENCE_THRESHOLD = 0.75

# ============================================================
# CLASS NAMES
# IMPORTANT: ORDER MUST MATCH TRAINING ORDER
# ============================================================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# ============================================================
# DOWNLOAD MODEL FROM AWS / S3
# ============================================================
from  helper_functions import download_directory
download_directory(BUCKET_NAME,S3_PREFIX, local_directory)
# ============================================================
# IMAGE PREPROCESSING
# ============================================================
from helper_functions import remove_bottom_annotation
train_eval_transform = transforms.Compose([
    transforms.Lambda(
        partial(remove_bottom_annotation, pixels=90)
    ),
    ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms(),
])

# ============================================================
# LOAD MODEL
# ============================================================
from helper_functions import create_ConvNeXt_Tiny_Weights_model
ConvNeXt_Tiny, _ = create_ConvNeXt_Tiny_Weights_model(num_classes=18,seed=42)

ConvNeXt_Tiny.load_state_dict(
    torch.load(local_directory+"/"+S3_PREFIX+MODEL_FILENAME, map_location="cpu", weights_only=True)
)
ConvNeXt_Tiny.eval()

## Put convext_tiny on cpu
ConvNeXt_Tiny.to("cpu") 

# Check the device
next(iter(ConvNeXt_Tiny.parameters())).device

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_image(image):
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """

    # Make sure image is RGB
    image = image.convert("RGB")
    # Transform the target image and add a batch dimension and send to device
    image = train_eval_transform(image).unsqueeze(0).to("cpu")

    # Put model into evaluation mode and turn on inference mode
    ConvNeXt_Tiny.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(ConvNeXt_Tiny(image), dim=1)

    confidence, predicted_index = torch.max(pred_probs,dim=1)
    predicted_index = predicted_index.item()
    confidence = confidence.item()

    predicted_class = class_names[predicted_index]

    # Also return top 3 predictions
    top_probabilities, top_indices = torch.topk(pred_probs,k=3,dim=1)

    top3 = []
    for probability, index in zip(top_probabilities[0],top_indices[0]):
        top3.append({
            "Class": class_names[index.item()],
            "Probability": probability.item()
        })
    return predicted_class, confidence, top3
    
# ============================================================
# STREAMLIT APP
# ============================================================
st.title("Rock Classification Model")

st.write("ConvNeXt-Tiny 18-Class Carbonate Rock Classifier")

# ============================================================
# SELECT PREDICTION TYPE
# ============================================================
st.header("2. Prediction")
prediction_mode = st.radio("Choose prediction mode",
    [
        "Single Image",
        "Multiple Images",
        "Local Test Folder",
        "URL"
    ]
)

if prediction_mode == "Single Image":
    from helper_functions import single_image_prediction
    single_image_prediction(predict_image, CONFIDENCE_THRESHOLD)
elif prediction_mode == "Multiple Images":
    from helper_functions import multiple_images_prediction
    multiple_images_prediction(predict_image, CONFIDENCE_THRESHOLD)
elif prediction_mode == "Local Test Folder":
    from helper_functions import local_test_prediction
    local_test_prediction(predict_image, CONFIDENCE_THRESHOLD)
elif prediction_mode == "URL":
    from helper_functions import url_prediction
    url_prediction(predict_image, CONFIDENCE_THRESHOLD)