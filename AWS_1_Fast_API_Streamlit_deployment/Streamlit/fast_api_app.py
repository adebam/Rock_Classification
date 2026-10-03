###########################################
#
#         Design POST REQUEST
#
###########################################
import json
import requests
import torch
import boto3

from functools import partial
from io import BytesIO
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from data_model import (ImageDataInput,ClassPrediction,ImageDataOutput,BatchImageDataOutput,S3ImageDataInput,S3BatchImageDataInput)
import uvicorn
import logging



app = FastAPI()
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

model_path = Path(local_directory) / S3_PREFIX / MODEL_FILENAME
if not model_path.is_file():
    download_directory(BUCKET_NAME, S3_PREFIX, local_directory)
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


@app.get("/")
def read_root():
    return "Hello! I am up!!!"

###################
#URL PREDICTION
###################
@app.post("/api/v1/get_rock_classification",response_model=BatchImageDataOutput,)
def rock_classification(data: ImageDataInput):
    results = []

    for image_url in data.url:
        url = str(image_url)

        # Download the image.
        try:
            with requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "RockClassifier/1.0", "Accept": "image/*"},
            ) as response:
                response.raise_for_status()
                image_bytes = response.content
        except requests.RequestException as exc:
            raise HTTPException(status_code=400,detail=f"Could not download image: {url}",
            ) from exc

        # Decode the downloaded bytes into a PIL image.
        try:
            with Image.open(BytesIO(image_bytes)) as source_image:
                image = source_image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"URL did not provide a readable image: {url}",
            ) from exc

        # preprocessing removes the bottom 90 pixels.
        if image.height <= 90:
            raise HTTPException(
                status_code=422,
                detail=f"Image must be taller than 90 pixels: {url}",
            )

        # Measure the existing prediction function.
        start = perf_counter()
        predicted_class, confidence, top3 = predict_image(image)
        prediction_time = perf_counter() - start

        # Package this image's prediction.
        result = ImageDataOutput(
            image_source=url,
            predicted_class=predicted_class,
            confidence=confidence,
            accepted=confidence >= CONFIDENCE_THRESHOLD,
            top3=[
                ClassPrediction(
                    label=item["Class"],
                    score=item["Probability"],
                )
                for item in top3
            ],
            prediction_time=prediction_time,
        )

        results.append(result)

    # Return every image's result and the total prediction time.
    return BatchImageDataOutput(
        total_prediction_time=sum(
            result.prediction_time for result in results
        ),
        results=results,
    )

###################
# SINGLE IMAGE
###################
from fastapi import UploadFile, File
@app.post(
    "/api/v1/rock_classification/single-image",
    response_model=ImageDataOutput,
)
async def single_image_rock_classification(
    file: UploadFile = File(...)
):

    # Read uploaded image bytes
    image_bytes = await file.read()

    # Convert bytes into PIL image
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file is not a readable image: {file.filename}",
        ) from exc

    # preprocessing removes bottom 90 pixels
    if image.height <= 90:
        raise HTTPException(
            status_code=422,
            detail=f"Image must be taller than 90 pixels: {file.filename}",
        )

    # Prediction
    start = perf_counter()
    predicted_class, confidence, top3 = predict_image(image)
    prediction_time = perf_counter() - start

    # Package result
    result = ImageDataOutput(
        image_source=file.filename,
        predicted_class=predicted_class,
        confidence=confidence,
        accepted=confidence >= CONFIDENCE_THRESHOLD,
        top3=[
            ClassPrediction(
                label=item["Class"],
                score=item["Probability"],
            )
            for item in top3
        ],
        prediction_time=prediction_time,
    )
    return result

######################################
# MULTIPLE IMAGES &local_test_prediction
######################################
from typing import Annotated
from fastapi import File, UploadFile
from pydantic import WithJsonSchema

SwaggerUploadFile = Annotated[
    UploadFile,
    WithJsonSchema({"type": "string", "format": "binary"}),
]
@app.post("/api/v1/rock_classification/multiple-images",response_model=BatchImageDataOutput,)
async def multiple_image_rock_classification(files: Annotated[list[SwaggerUploadFile], File()],):

    results = []
    for file in files:
        image_bytes = await file.read()
        try:
            with Image.open(BytesIO(image_bytes)) as source_image:
                image = source_image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Uploaded file is not a readable image: {file.filename}",
            ) from exc
        if image.height <= 90:
            raise HTTPException(
                status_code=422,
                detail=f"Image must be taller than 90 pixels: {file.filename}",
            )

        start = perf_counter()

        predicted_class, confidence, top3 = predict_image(image)

        prediction_time = perf_counter() - start

        result = ImageDataOutput(
            image_source=file.filename,
            predicted_class=predicted_class,
            confidence=confidence,
            accepted=confidence >= CONFIDENCE_THRESHOLD,
            top3=[
                ClassPrediction(
                    label=item["Class"],
                    score=item["Probability"],
                )
                for item in top3
            ],
            prediction_time=prediction_time,
        )

        results.append(result)

    return BatchImageDataOutput(
        total_prediction_time=sum(
            result.prediction_time for result in results
        ),
        results=results,
    )

###################
# S3 SINGLE IMAGE
###################
import logging

@app.post(
    "/api/v1/rock_classification/s3-image",
    response_model=ImageDataOutput,
)
def s3_image_rock_classification(data: S3ImageDataInput):
    # Download image from S3
    try:
        s3 = boto3.client("s3")
        response = s3.get_object(
            Bucket=data.bucket_name,
            Key=data.key,
        )

        try:
            image_bytes = response["Body"].read()
        finally:
            response["Body"].close()

    except Exception as exc:
        logging.exception("S3 download failed")
        raise HTTPException(
            status_code=400,
            detail=f"Could not download S3 object: {data.key}",
        ) from exc

    # Convert bytes to PIL image
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"S3 object is not a readable image: {data.key}",
        ) from exc

    # Preprocessing removes the bottom 90 pixels
    if image.height <= 90:
        raise HTTPException(
            status_code=422,
            detail=f"Image must be taller than 90 pixels: {data.key}",
        )

    # Prediction
    start = perf_counter()
    predicted_class, confidence, top3 = predict_image(image)
    prediction_time = perf_counter() - start

    return ImageDataOutput(
        image_source=f"s3://{data.bucket_name}/{data.key}",
        predicted_class=predicted_class,
        confidence=confidence,
        accepted=confidence >= CONFIDENCE_THRESHOLD,
        top3=[
            ClassPrediction(
                label=item["Class"],
                score=item["Probability"],
            )
            for item in top3
        ],
        prediction_time=prediction_time,
    )
    
###################
# MULTIPLE S3 IMAGES
###################

@app.post(
    "/api/v1/rock_classification/multiple-s3-images",
    response_model=BatchImageDataOutput,
)
def multiple_s3_image_rock_classification(data: S3BatchImageDataInput):
    s3 = boto3.client("s3")
    results = []
    for key in data.keys:
        # ---------------------------------------------
        # DOWNLOAD IMAGE FROM S3
        # ---------------------------------------------
        try:
            response = s3.get_object(Bucket=data.bucket_name,Key=key,)
            try:
                image_bytes = response["Body"].read()
            finally:
                response["Body"].close()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not download S3 object: {key}. "
                    f"Error: {str(exc)}"
                ),
            ) from exc
        # ---------------------------------------------
        # CONVERT TO PIL
        # ---------------------------------------------
        try:
            with Image.open(BytesIO(image_bytes) ) as source_image:image = source_image.convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not download S3 object: {key}. "
                    f"Error: {str(exc)}"
                ),
            ) from exc
        # ---------------------------------------------
        # CHECK IMAGE HEIGHT
        # ---------------------------------------------
        if image.height <= 90:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Image must be taller than "
                    f"90 pixels: {key}"
                ),
            )
        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------
        start = perf_counter()
        predicted_class, confidence, top3 = (predict_image(image))
        prediction_time = (perf_counter() - start)
        # ---------------------------------------------
        # PACKAGE RESULT
        # ---------------------------------------------
        result = ImageDataOutput(
            image_source=(f"s3://{data.bucket_name}/{key}"),
            predicted_class=predicted_class,
            confidence=confidence,
            accepted=(confidence >= CONFIDENCE_THRESHOLD),
            top3=[
                ClassPrediction(
                    label=item["Class"],
                    score=item["Probability"],
                )
                for item in top3
            ],
            prediction_time=prediction_time,
        )
        results.append(result)
    # ---------------------------------------------
    # RETURN BATCH
    # ---------------------------------------------
    return BatchImageDataOutput(
        total_prediction_time=sum(
            result.prediction_time
            for result in results
        ),
        results=results,
    )    
    
 # connects fast api to streamlit   
if __name__=="__main__":
    uvicorn.run(app="fast_api_app:app", port=8000, reload=True, host="0.0.0.0")