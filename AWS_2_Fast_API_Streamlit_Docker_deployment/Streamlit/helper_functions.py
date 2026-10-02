from PIL import Image
from torchvision import models, transforms
from functools import partial
from torchvision.models import ConvNeXt_Tiny_Weights
import torch
import torchvision
from torch import nn
import streamlit as st
import pandas as pd
import boto3
import requests
from io import BytesIO
from PIL import Image
from time import perf_counter
from data_model import BatchImageDataOutput, ClassPrediction, ImageDataOutput
import os
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

def download_directory(BUCKET_NAME,S3_PREFIX, local_directory):
    import os
    import boto3
    session =boto3.Session()
    s3  = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=BUCKET_NAME,
        Prefix=S3_PREFIX
    ):
        if "Contents" not in page:
            print(f"No files found under {S3_PREFIX}")
            return

        for obj in page["Contents"]:
            s3_key = obj["Key"]
            # skip folder placeholder objects
            if s3_key.endswith("/"):
                continue
            # keep entire S3 path
            local_path = os.path.join( local_directory,s3_key)
            # create folders if needed
            os.makedirs(os.path.dirname(local_path),exist_ok=True)
            print(f"Downloading: {s3_key}")
            print(f"          to: {local_path}")
            s3.download_file(BUCKET_NAME,s3_key,local_path)


def remove_bottom_annotation(image, pixels):
    """
    this function can be used to remove the bottom part. It removes the ruler.
    """
    width, height = image.size
    if not 0 <= pixels < height:
        raise ValueError(f"pixels must be between 0 and {height - 1}, got {pixels}")
    return image.crop((0, 0, width, height - pixels))

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


def single_image_prediction():
    import requests

    uploaded_file = st.file_uploader("Upload rock image",type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file,caption=uploaded_file.name,width=500)
        if st.button("Predict"):
            api_url = (
                f"{API_BASE_URL}/"
                "api/v1/rock_classification/single-image"
            )

            files = {"file": (uploaded_file.name,uploaded_file.getvalue(),uploaded_file.type)}
            try:
                response = requests.post(api_url,files=files,timeout=30)
                response.raise_for_status()
                result = response.json()

                predicted_class = result["predicted_class"]
                confidence = result["confidence"]
                accepted = result["accepted"]
                top3 = result["top3"]
                prediction_time = result["prediction_time"]

                st.subheader("Prediction")
                st.write(f"### {predicted_class}")
                st.write(f"Confidence: **{confidence:.2%}**")

                if accepted:
                    st.success("High-confidence prediction")
                else:
                    st.warning("Low-confidence prediction")

                st.write(f"Prediction time: "f"**{prediction_time:.3f} seconds**")
                st.subheader("Top 3 Predictions")
                top3_df = pd.DataFrame(top3)
                top3_df["score"] = (top3_df["score"] * 100).round(2)
                top3_df = top3_df.rename(columns={"label": "Class","score": "Probability (%)"})
                st.dataframe(top3_df,use_container_width=True)
            except requests.RequestException as e:
                st.error(
                    f"FastAPI request failed: {e}"
                )

def multiple_images_prediction():
    uploaded_files = st.file_uploader(
        "Upload multiple test images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)} images**")
        if st.button("Run Batch Prediction"):
            api_url = (
                f"{API_BASE_URL}/"
                "api/v1/rock_classification/multiple-images"
            )
            # Prepare files for FastAPI
            files = []
            for uploaded_file in uploaded_files:
                files.append(
                    (
                        "files",
                        (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    )
                )
            try:
                response = requests.post(
                    api_url,
                    files=files,
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                # -----------------------------------------
                # Batch information
                # -----------------------------------------
                st.write(f"Total prediction time: "f"**{data['total_prediction_time']:.3f} seconds**")
                # -----------------------------------------
                # Build results table
                # -----------------------------------------
                results = []
                for result in data["results"]:
                    results.append({
                        "Filename": result["image_source"],
                        "Prediction": result["predicted_class"],
                        "Confidence": result["confidence"],
                        "Accepted": result["accepted"],
                    })
                results_df = pd.DataFrame(results)
                results_df["Confidence"] = (results_df["Confidence"] * 100).round(2)

                st.subheader("Results")
                st.dataframe(results_df,use_container_width=True)
                # -----------------------------------------
                # Coverage
                # -----------------------------------------
                accepted = results_df["Accepted"].sum()
                total = len(results_df)
                coverage = accepted / total

                st.write(f"### Coverage: {coverage:.2%}")
                st.write(f"Accepted images: **{accepted}/{total}**")

            except requests.RequestException as e:
                st.error(f"FastAPI request failed: {e}")
            st.markdown(
            "> **Note:**  \n"
            "> **Prediction** — *The predicted class of the image.*  \n"
            "> **Confidence** — *The probability assigned to the predicted class "
            "(the class with the highest predicted probability).*  \n"
            "> **Accepted** — *A prediction is accepted if its confidence is greater than "
            "or equal to the minimum confidence threshold (75%).*  \n"
        )

def local_test_prediction():
    folder_path = st.text_input("Enter local test folder path")

    if folder_path and st.button("Run Folder Prediction"):
        if not os.path.isdir(folder_path):
            st.error("Folder does not exist.")
            return

        image_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(folder_path, filename))

        if not image_files:
            st.warning("No images found.")
            return

        st.write(f"Found **{len(image_files)} images**")

        api_url = (
            f"{API_BASE_URL}/"
            "api/v1/rock_classification/multiple-images"
        )
        files = []
        opened_files = []
        try:
            for image_path in image_files:
                f = open(image_path, "rb")
                opened_files.append(f)
                files.append(
                    (
                        "files",
                        (
                            os.path.basename(image_path),
                            f,
                            "image/jpeg",
                        )
                    )
                )

            response = requests.post( api_url,files=files,timeout=120)
            response.raise_for_status()
            data = response.json()
            results = []

            for result in data["results"]:
                results.append({
                    "Filename": result["image_source"],
                    "Prediction": result["predicted_class"],
                    "Confidence": result["confidence"],
                    "Accepted": result["accepted"],
                })
            results_df = pd.DataFrame(results)
            results_df["Confidence"] = (results_df["Confidence"] * 100).round(2)

            st.subheader("Results")
            st.dataframe(results_df,use_container_width=True)
            accepted = results_df["Accepted"].sum()
            total = len(results_df)
            coverage = accepted / total

            st.write(f"### Coverage: {coverage:.2%}")
            st.write(f"Accepted images: **{accepted}/{total}**")

        except requests.RequestException as e:
            st.error(f"FastAPI request failed: {e}")

        finally:
            for f in opened_files:
                f.close()
            st.markdown(
                "> **Note:**  \n"
                "> **Prediction** — *The predicted class of the image.*  \n"
                "> **Confidence** — *The probability assigned to the predicted class "
                "(the class with the highest predicted probability).*  \n"
                "> **Accepted** — *A prediction is accepted if its confidence is greater than "
                "or equal to the minimum confidence threshold (75%).*  \n"
                "> **Threshold** — *The minimum confidence required for a prediction to be accepted.*  \n"
                "> **Coverage** — *The percentage of images accepted: "
                "accepted images / total images × 100.*"
            )

def url_prediction():
    url = st.text_input("Enter image URL")

    if url:
        st.image(url, caption="Image from URL")
        if st.button("Predict"):
            api_url = (
                f"{API_BASE_URL}/"
                "api/v1/get_rock_classification"
            )
            payload = {"url": [url]}
            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                st.subheader("FastAPI Response in a nice table")
                result = data["results"][0]
                predicted_class = result["predicted_class"]
                confidence = result["confidence"]
                accepted = result["accepted"]
                top3 = result["top3"]
                prediction_time = result["prediction_time"]
                
                st.subheader("Prediction")
                st.write(f"### {predicted_class}")
                st.write(f"Confidence: **{confidence:.2%}**")
                if accepted:
                    st.success("High-confidence prediction")
                else:
                    st.warning("Low-confidence prediction")
                st.write(f"Prediction time: **{prediction_time:.3f} seconds**")
                
                st.subheader("Top 3 Predictions")
                top3_df = pd.DataFrame(top3)
                top3_df["score"] = (top3_df["score"] * 100).round(2)
                top3_df = top3_df.rename(
                    columns={
                        "label": "Class",
                        "score": "Probability (%)"
                    }
                )
                st.dataframe(top3_df,use_container_width=True)
            
            except requests.RequestException as e:
                st.error(f"API request failed: {e}")



def s3_prediction(
    BUCKET_NAME="super-dayo-409400548716",
    image_prefix="data/Rock-classification/test/",
):

    s3 = boto3.client("s3")
    image_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    # ---------------------------------------------------------
    # LIST IMAGES FROM S3
    # ---------------------------------------------------------
    for page in paginator.paginate(
        Bucket=BUCKET_NAME,
        Prefix=image_prefix,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                image_keys.append(key)
    if not image_keys:
        st.info("No images found in this S3 folder.")
        return
    # ---------------------------------------------------------
    # SELECT IMAGE
    # ---------------------------------------------------------
    selected_key = st.selectbox("Choose an image from S3",sorted(image_keys),)
    # ---------------------------------------------------------
    # DOWNLOAD IMAGE ONLY FOR STREAMLIT PREVIEW
    # ---------------------------------------------------------
    response = s3.get_object(Bucket=BUCKET_NAME,Key=selected_key,)
    try:
        image_bytes = response["Body"].read()
    finally:
        response["Body"].close()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(image,caption=selected_key)
    # ---------------------------------------------------------
    # SEND S3 LOCATION TO FASTAPI
    # ---------------------------------------------------------
    if st.button("Predict S3 image"):
        api_url = (
            f"{API_BASE_URL}/"
            "api/v1/rock_classification/s3-image"
        )
        payload = {
            "bucket_name": BUCKET_NAME,
            "key": selected_key,
        }
        try:
            response = requests.post(
                api_url,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            # -------------------------------------------------
            # GET VALUES RETURNED BY FASTAPI
            # -------------------------------------------------
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            accepted = result["accepted"]
            top3 = result["top3"]
            prediction_time = result["prediction_time"]
            # -------------------------------------------------
            # DISPLAY RESULTS
            # -------------------------------------------------
            st.subheader("Prediction")
            st.write(f"### {predicted_class}")
            st.write(f"Confidence: **{confidence:.2%}**")
            if accepted:
                st.success("High-confidence prediction")
            else:
                st.warning("Low-confidence prediction")
            st.write(f"Prediction time: "
                f"**{prediction_time:.3f} seconds**"
            )
            # -------------------------------------------------
            # TOP 3
            # -------------------------------------------------
            st.subheader("Top 3 Predictions")
            top3_df = pd.DataFrame(top3)
            top3_df["score"] = (top3_df["score"] * 100).round(2)
            top3_df = top3_df.rename(
                columns={
                    "label": "Class",
                    "score": "Probability (%)",
                }
            )
            st.dataframe(top3_df,use_container_width=True)
        except requests.RequestException as e:
            st.error(f"FastAPI request failed: {e}")


def s3_multi_image_prediction(
    BUCKET_NAME="super-dayo-409400548716",
    image_prefix="data/Rock-classification/test/",
confidence_threshold=0.75):
    import requests
    s3 = boto3.client("s3")
    image_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    # ---------------------------------------------------------
    # LIST S3 IMAGES
    # ---------------------------------------------------------
    for page in paginator.paginate(Bucket=BUCKET_NAME,Prefix=image_prefix,):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                image_keys.append(key)

    st.write(f"Images found in S3: **{len(image_keys)}**")
    if not image_keys:
        st.error(
            f"No images found under:\n\n"
            f"s3://{BUCKET_NAME}/{image_prefix}"
        )
        return
    image_keys = sorted(image_keys)
    # ---------------------------------------------------------
    # SELECT IMAGES
    # ---------------------------------------------------------
    select_all = st.checkbox("Select all S3 images")
    if select_all:
        selected_keys = image_keys
    else:
        selected_keys = st.multiselect("Choose images from S3",image_keys,)
    if selected_keys:
        st.write(f"Selected **{len(selected_keys)} images**")
        if st.button("Run S3 Batch Prediction"):
            api_url = (
                f"{API_BASE_URL}/"
                "api/v1/rock_classification/"
                "multiple-s3-images"
            )
            payload = {"bucket_name": BUCKET_NAME,"keys": selected_keys,}
            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                # ---------------------------------------------
                # RESULTS TABLE
                # ---------------------------------------------
                results = []
                for result in data["results"]:
                    image_source = result["image_source"]
                    results.append({
                        "Filename": image_source.split("/")[-1],
                        "S3 Folder": image_source.split("/")[-2],
                        "Prediction": result["predicted_class"],
                        "Confidence": result["confidence"],
                        "Accepted": result["accepted"],
                    })

                results_df = pd.DataFrame(results)
                results_df["Confidence"] = (results_df["Confidence"] * 100).round(2)
                st.subheader("Results")
                st.dataframe(results_df,use_container_width=True,)
                # ---------------------------------------------
                # TOTAL PREDICTION TIME
                # ---------------------------------------------
                st.write(f"Total prediction time: "f"**{data['total_prediction_time']:.3f} seconds**")
                # ---------------------------------------------
                # COVERAGE
                # ---------------------------------------------
                accepted = results_df["Accepted"].sum()
                total = len(results_df)
                coverage = accepted / total

                st.write(f"### Coverage: {coverage:.2%}")
                st.write(f"Accepted images: **{accepted}/{total}**")
            except requests.RequestException as e:
                st.error(f"FastAPI request failed: {e}")
            st.write(f"Accepted images: **{accepted}/{total}**")
            st.markdown(
                "> **Note:**  \n"
                "> **Prediction** — *The predicted class of the image.*  \n"
                "> **Confidence** — *The probability assigned to the predicted class "
                "(the class with the highest predicted probability).*  \n"
                "> **Accepted** — *A prediction is accepted if its confidence is greater than "
                f"or equal to the minimum confidence threshold ({confidence_threshold:.0%}).*  \n"
                "> **Threshold** — *The minimum confidence required for a prediction to be accepted.*  \n"
                "> **Coverage** — *The percentage of images accepted: "
                "accepted images / total images × 100.*"
            )