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
from io import BytesIO
from PIL import Image

def download_model():

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    s3.download_file(BUCKET_NAME,S3_MODEL_KEY,LOCAL_MODEL_PATH)

    return LOCAL_MODEL_PATH

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



def single_image_prediction(predict_fn, confidence_threshold):
    uploaded_file = st.file_uploader(
        "Upload rock image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption=uploaded_file.name,
            width=500
        )
        if st.button("Predict"):
            predicted_class, confidence, top3 = predict_fn(image)
            st.subheader("Prediction")
            st.write(
                f"### {predicted_class}"
            )
            st.write(
                f"Confidence: **{confidence:.2%}**"
            )
            # -----------------------------------------------
            # Confidence threshold
            # -----------------------------------------------
            if confidence >= confidence_threshold:
                st.success(
                    f"High-confidence prediction "
                    f"(≥ {confidence_threshold:.0%})"
                )
            else:
                st.warning(
                    f"Low-confidence prediction "
                    f"(< {confidence_threshold:.0%})"
                )
            st.subheader("Top 3 Predictions")
            top3_df = pd.DataFrame(top3)
            top3_df["Probability"] = (top3_df["Probability"] * 100).round(2)
            st.dataframe(top3_df,use_container_width=True)

def multiple_images_prediction(predict_fn, confidence_threshold):
    uploaded_files = st.file_uploader(
    "Upload multiple test images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
    )
    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)} images**")
        if st.button("Run Batch Prediction"):
            results = []
            progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                predicted_class, confidence, _ = predict_fn(image)
                results.append({
                    "Filename": uploaded_file.name,
                    "Prediction": predicted_class,
                    "Confidence": confidence,
                    "Accepted": confidence >= confidence_threshold
                })
                progress.progress((i + 1) / len(uploaded_files))
            results_df = pd.DataFrame(results)
            results_df["Confidence"] = (
                results_df["Confidence"] * 100).round(2)
            st.subheader("Results")
            st.dataframe(
                results_df,
                use_container_width=True)
            st.markdown(
            "> **Note:**  \n"
            "> **Prediction** — *The predicted class of the image.*  \n"
            "> **Confidence** — *The probability assigned to the predicted class "
            "(the class with the highest predicted probability).*  \n"
            "> **Accepted** — *A prediction is accepted if its confidence is greater than "
            "or equal to the minimum confidence threshold (75%).*  \n"
        )

def local_test_prediction(predict_fn, confidence_threshold):
    uploaded_files = st.file_uploader(
        "Upload multiple test images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.write( f"Selected **{len(uploaded_files)} images**")
        if st.button("Run Batch Prediction"):
            results = []
            progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                predicted_class, confidence, _ = predict_fn(image)
                results.append({
                    "Filename": uploaded_file.name,
                    "Prediction": predicted_class,
                    "Confidence": confidence,
                    "Accepted": confidence >= confidence_threshold
                })
                progress.progress(
                    (i + 1) / len(uploaded_files)
                )
            results_df = pd.DataFrame(results)
            results_df["Confidence"] = (results_df["Confidence"] * 100).round(2)
            st.subheader("Results")
            st.dataframe(results_df,use_container_width=True)
            # -----------------------------------------------
            # Coverage
            # -----------------------------------------------
            accepted = results_df["Accepted"].sum()
            total = len(results_df)
            coverage = accepted / total
            st.write(
                f"### Coverage at "
                f"{confidence_threshold:.0%} threshold: "
                f"{coverage:.2%}"
            )
            st.write(f"Accepted images: **{accepted}/{total}**")
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

def url_prediction(predict_fn, confidence_threshold):
    import requests
    from io import BytesIO
    from PIL import Image
    url = st.text_input("Enter image URL")
    if url:
        try:
            response = requests.get(url,timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image,caption="Image from URL")
            if st.button("Predict"):
                predicted_class, confidence, top3 = predict_fn(image)

                st.subheader("Prediction")
                st.write(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.2%}")
                # -----------------------------------------------
                # Confidence threshold
                # -----------------------------------------------
                if confidence >= confidence_threshold:
                    st.success(
                        f"High-confidence prediction "
                        f"(≥ {confidence_threshold:.0%})"
                    )
                else:
                    st.warning(
                        f"Low-confidence prediction "
                        f"(< {confidence_threshold:.0%})"
                    )
                st.subheader("Top 3 Predictions")
                top3_df = pd.DataFrame(top3)
                top3_df["Probability"] = (top3_df["Probability"] * 100).round(2)
                st.dataframe(top3_df,use_container_width=True)
        except Exception as e:
            st.error(f"Could not load image from URL: {e}") 



def s3_prediction(predict_fn,confidence_threshold,BUCKET_NAME="super-dayo-409400548716",image_prefix="data/Rock-classification/test/",):
    s3 = boto3.client("s3")
    image_keys = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET_NAME,Prefix=image_prefix,):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".jpg", ".jpeg", ".png")):
                image_keys.append(key)

    if not image_keys:
        st.info("No images found in this S3 folder.")
        return

    selected_key = st.selectbox("Choose an image from S3",sorted(image_keys),)

    response = s3.get_object(Bucket=BUCKET_NAME,Key=selected_key,)
    try:
        image_bytes = response["Body"].read()
    finally:
        response["Body"].close()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption=selected_key)
    if st.button("Predict S3 image"):
        predicted_class, confidence, top3 = predict_fn(image)
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2%}")

        # Confidence threshold
        if confidence >= confidence_threshold:
            st.success(
                f"High-confidence prediction "
                f"(≥ {confidence_threshold:.0%})"
            )
        else:
            st.warning(
                f"Low-confidence prediction "
                f"(< {confidence_threshold:.0%})"
            )

        st.subheader("Top 3 Predictions")
        top3_df = pd.DataFrame(top3)
        top3_df["Probability"] = (
            top3_df["Probability"] * 100
        ).round(2)
        st.dataframe(top3_df, use_container_width=True)


def s3_multi_image_prediction(
    predict_fn,
    confidence_threshold,
    BUCKET_NAME="super-dayo-409400548716",
    image_prefix="data/Rock-classification/test/",
):

    s3 = boto3.client("s3")

    st.write("S3 Bucket:", BUCKET_NAME)
    st.write("S3 Prefix:", image_prefix)

    image_keys = []

    paginator = s3.get_paginator("list_objects_v2")

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

    st.write(f"Images found in S3: **{len(image_keys)}**")

    if not image_keys:
        st.error(
            f"No images found under:\n\n"
            f"s3://{BUCKET_NAME}/{image_prefix}"
        )
        return

    image_keys = sorted(image_keys)

    # ---------------------------------------------------------
    # SELECT ALL OR SELECT SPECIFIC IMAGES
    # ---------------------------------------------------------

    select_all = st.checkbox(
        "Select all S3 images"
    )

    if select_all:

        selected_keys = image_keys

    else:

        selected_keys = st.multiselect(
            "Choose images from S3",
            image_keys,
        )

    if selected_keys:

        st.write(
            f"Selected **{len(selected_keys)} images**"
        )

        if st.button("Run S3 Batch Prediction"):

            results = []

            progress = st.progress(0)

            for i, selected_key in enumerate(selected_keys):

                # ---------------------------------------------
                # Download image from S3
                # ---------------------------------------------

                response = s3.get_object(
                    Bucket=BUCKET_NAME,
                    Key=selected_key,
                )

                try:

                    image_bytes = response["Body"].read()

                finally:

                    response["Body"].close()

                # ---------------------------------------------
                # Convert to PIL image
                # ---------------------------------------------

                image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

                # ---------------------------------------------
                # Prediction
                # ---------------------------------------------

                predicted_class, confidence, _ = predict_fn(
                    image
                )

                results.append({

                    "Filename": selected_key.split("/")[-1],

                    "S3 Folder": selected_key.split("/")[-2],

                    "Prediction": predicted_class,

                    "Confidence": confidence,

                    "Accepted":
                        confidence >= confidence_threshold
                })

                progress.progress(
                    (i + 1) / len(selected_keys)
                )

            # ================================================
            # RESULTS
            # ================================================

            results_df = pd.DataFrame(results)

            results_df["Confidence"] = (
                results_df["Confidence"] * 100
            ).round(2)

            st.subheader("Results")

            st.dataframe(
                results_df,
                use_container_width=True
            )

            # ================================================
            # COVERAGE
            # ================================================

            accepted = results_df["Accepted"].sum()

            total = len(results_df)

            coverage = accepted / total

            st.write(
                f"### Coverage at "
                f"{confidence_threshold:.0%} threshold: "
                f"{coverage:.2%}"
            )

            st.write(
                f"Accepted images: **{accepted}/{total}**"
            )

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