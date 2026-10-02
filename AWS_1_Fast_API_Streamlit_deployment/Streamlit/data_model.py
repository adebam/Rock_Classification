from pydantic import BaseModel, Field, HttpUrl


class ImageDataInput(BaseModel):
    url: list[HttpUrl] = Field(min_length=1)


class ClassPrediction(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class ImageDataOutput(BaseModel):
    model_name: str = "ConvNeXt-Tiny-18-Class"
    image_source: str  # Filename, S3 path, or URL
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    accepted: bool
    top3: list[ClassPrediction]
    prediction_time: float = Field(ge=0)  # Seconds


class BatchImageDataOutput(BaseModel):
    """Batch results with summed prediction time, excluding image loading."""

    total_prediction_time: float = Field(
        ge=0,
        description="Sum of per-image prediction times in seconds; excludes loading and downloads",
    )
    results: list[ImageDataOutput]

class S3ImageDataInput(BaseModel):
    bucket_name: str
    key: str

class S3BatchImageDataInput(BaseModel):
    bucket_name: str
    keys: list[str] = Field(min_length=1)