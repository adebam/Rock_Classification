import streamlit as st
import os
import torch
import torch.nn as nn
import pandas as pd
import json
import boto3
from torchvision import models, transforms
from functools import partial


# ============================================================
# STREAMLIT APP
# ============================================================
st.title("Rock Classification Model")

st.write("ConvNeXt-Tiny 18-Class Carbonate Rock Classifier")

# ============================================================
# SELECT PREDICTION TYPE
# ============================================================
st.markdown("##### *How to choose*")
st.write("choose *s3 Images* to analyze one image from Amazon s3 bucket")
st.write("choose *Multiple s3 Images* to analyze multiple images from Amazon s3 bucket")
st.write("choose *Single Images* to analyze a single image uploaded from local computer")
st.write("choose *Multiple Images* to analyze a multiple images uploaded from local computer")
st.write("choose *url* to analyze one image from a link")

st.header("2. Prediction")
prediction_mode = st.radio("Choose prediction mode",
    [
        "s3 Images",
        "Multiple s3 Images",
        "Single Image",
        "Multiple Images",
        #"Local Test Folder",
        "URL"
    ]
)

if prediction_mode == "Single Image":
    from helper_functions import single_image_prediction
    single_image_prediction()
elif prediction_mode == "Multiple Images":
    from helper_functions import multiple_images_prediction
    multiple_images_prediction()
elif prediction_mode == "Local Test Folder":
    from helper_functions import local_test_prediction
    local_test_prediction()
elif prediction_mode == "URL":
    from helper_functions import url_prediction
    url_prediction()
elif prediction_mode == "s3 Images":
    from helper_functions import s3_prediction
    s3_prediction(BUCKET_NAME="super-dayo-409400548716",image_prefix="data/Rock-classification/test/")
elif prediction_mode == "Multiple s3 Images":
    from helper_functions import s3_multi_image_prediction
    s3_multi_image_prediction(BUCKET_NAME="super-dayo-409400548716",image_prefix="data/Rock-classification/test/",confidence_threshold=0.75)
