
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_awesome_barchart(df):
    # Sort values descending by percent for a clean pareto look
    df_sorted = df.sort_values('percent', ascending=False).reset_index(drop=True)

    # Set the style background
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the horizontal barplot
    sns.barplot(
        x='percent', 
        y='class', 
        data=df_sorted, 
        palette="viridis", # A highly readable and modern color palette
        ax=ax
    )

    # Add data labels to the end of each bar
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.15,                   # Offset slightly to the right of the bar
            p.get_y() + p.get_height() / 2, # Center vertically on the bar
            f'{width:.2f}%', 
            ha='left', 
            va='center', 
            fontsize=10,
            fontweight='bold',
            color='#333333'
        )

    # Formatting aesthetics
    ax.set_title('Distribution of Rock Class by Percentage', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Class', fontsize=14, fontweight='bold')

    # Remove top and right borders for a cleaner look
    sns.despine(left=True, bottom=True)

    # Extend x-axis slightly to make room for the text labels
    ax.set_xlim(0, df_sorted['percent'].max() + 2)

    plt.tight_layout()
    plt.show()

import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def plot_random_image(image_path, seed=None):
    """
    Randomly selects and displays one image from a directory.

    Args:
        image_path: Path to a directory containing images.
        seed: Optional random seed for reproducible image selection.
    """

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    image_path = Path(image_path)
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    # Get all valid image paths
    image_path_list = [
        path for path in image_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in valid_extensions
        and ".ipynb_checkpoints" not in path.parts
    ]
    # Make sure images were found
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in: {image_path}")
    # Select random image
    random_image_path = random.choice(image_path_list)
    # Get class from parent folder
    image_class = random_image_path.parent.stem
    # Open image
    img = Image.open(random_image_path)
    # Convert image to NumPy array
    img_as_array = np.asarray(img)
    # Plot
    plt.figure(figsize=(10, 7))
    plt.imshow(img_as_array)
    plt.title(
        f"Image class: {image_class} | "
        f"Image shape: {img_as_array.shape} "
        f"-> [height, width, color_channels]"
    )

    plt.axis("off")
    plt.show()
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image


def plot_transformed_images(
    image_paths,
    transform,
    mean,
    std,
    n: int = 3,
    seed = None
):
    """
    Plots original rock images beside their transformed versions.

    Randomly selects n image paths, applies the supplied PyTorch
    transformation, unstandardizes the transformed image for
    visualization, and plots the original and transformed images
    side by side.

    Args:
        image_paths: List of image paths.
        transform: PyTorch transformation pipeline to apply to each image.
        mean: RGB mean values used for image normalization.
        std: RGB standard deviation values used for image normalization.
        n: Number of random images to display. Default is 3.
        seed: Random seed used for reproducible image selection.
            Default is 42.

    Example usage:
        plot_transformed_images(
            image_paths=image_path_list,
            transform=train_transform,
            mean=mean,
            std=std,
            n=3,
            seed=42
        )
    """
    # Set random seeds
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Make sure n is not larger than number of available images
    n = min(n, len(image_paths))
    # Randomly select images
    random_image_paths = random.sample(image_paths, k=n)
    # Convert mean/std into shape [C, 1, 1]
    # so they can be applied to an image tensor [C, H, W]
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    for image_path in random_image_paths:
        # Load image
        with Image.open(image_path).convert("RGB") as image:
            # ----------------------------------------
            # Apply transformation
            # ----------------------------------------
            transformed_image = transform(image)
            # ----------------------------------------
            # Undo normalization for visualization
            # x_original = x_normalized * std + mean<abs
            # ----------------------------------------
            transformed_image_display = (transformed_image * std_tensor + mean_tensor)
            # Keep pixel values between 0 and 1
            transformed_image_display = transformed_image_display.clamp(0, 1)
            # Convert [C, H, W] -> [H, W, C]
            transformed_image_display = transformed_image_display.permute(1, 2, 0)
            # ----------------------------------------
            # Plot
            # ----------------------------------------
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # Original
            ax[0].imshow(image)
            ax[0].set_title(f"Original\nSize: {image.size}")
            ax[0].axis("off")
            # Transformed
            ax[1].imshow(transformed_image_display,interpolation="none")
            ax[1].set_title(
                f"Transformed\n"
                f"Size: {transformed_image.shape[1]} × "
                f"{transformed_image.shape[2]}"
            )
            ax[1].axis("off")
            # Class name comes from parent folder
            fig.suptitle(f"Class: {image_path.parent.name}",fontsize=16)
            plt.tight_layout()
            plt.show()
    return fig
import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(results, model_name):

    epochs = range(1, len(results["train_loss"]) + 1)

    metrics = [("loss", "Loss"),("acc", "Accuracy"),("f1", "F1 Score"),("precision", "Precision"),("recall", "Recall")]
    # -------------------------------------------------
    # Create 2 x 3 subplot figure
    # -------------------------------------------------
    fig, axes = plt.subplots(2,3,figsize=(18, 10))
    axes = axes.flatten()
    # -------------------------------------------------
    # Training / Validation metrics
    # -------------------------------------------------
    for ax, (key, title) in zip(axes, metrics):
        ax.plot(epochs,results[f"train_{key}"],marker="o",label=f"Train {title}")
        ax.plot(epochs,results[f"test_{key}"],marker="o",label=f"Validation {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Training vs Validation {title}")
        ax.legend()
        ax.grid(alpha=0.3)

    # -------------------------------------------------
    # Cumulative training time
    # -------------------------------------------------
    cumulative_time = np.cumsum(results["epoch_time"])
    ax = axes[5]
    ax.plot(epochs,cumulative_time,marker="o")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative Time (seconds)")
    ax.set_title("Cumulative Training Time")
    ax.grid(alpha=0.3)

    # -------------------------------------------------
    # Overall figure title
    # -------------------------------------------------
    fig.suptitle(
        f"{model_name} — Training Performance",
        fontsize=20,
        fontweight="bold"
    )

    # -------------------------------------------------
    # Adjust spacing
    # -------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchmetrics import ConfusionMatrix
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassRecall
)

from mlxtend.plotting import plot_confusion_matrix


def plot_model_diagnostics(
    model,
    train_dataloader,
    test_dataloader,
    class_names,
    model_name,
    device
):

    # =========================================================
    # 1. GET MODEL PREDICTIONS
    # =========================================================
    model.eval()

    y_preds = []
    y_true = []

    with torch.inference_mode():

        for X, y in test_dataloader:

            X = X.to(device)
            y = y.to(device)

            # Forward pass
            y_logits = model(X)

            # Predicted class
            y_pred = torch.argmax(y_logits, dim=1)

            # Store predictions and true labels
            y_preds.append(y_pred.cpu())
            y_true.append(y.cpu())


    # Combine all batches
    y_pred_tensor = torch.cat(y_preds)
    y_true_tensor = torch.cat(y_true)


    # =========================================================
    # 2. CONFUSION MATRIX
    # =========================================================
    confmat = ConfusionMatrix(
        num_classes=len(class_names),
        task="multiclass"
    )

    confmat_tensor = confmat(
        preds=y_pred_tensor,
        target=y_true_tensor
    )


    # =========================================================
    # 3. F1 SCORE FOR EACH CLASS
    # =========================================================
    f1_metric = MulticlassF1Score(
        num_classes=len(class_names),
        average=None
    )

    f1_scores = f1_metric(
        y_pred_tensor,
        y_true_tensor
    ).numpy()


    # =========================================================
    # 4. RECALL FOR EACH CLASS
    # =========================================================
    recall_metric = MulticlassRecall(
        num_classes=len(class_names),
        average=None
    )

    recall_scores = recall_metric(
        y_pred_tensor,
        y_true_tensor
    ).numpy()


    # =========================================================
    # 5. GET TRAINING CLASS COUNTS
    # =========================================================
    train_labels = [
        label
        for _, label in train_dataloader.dataset
    ]

    train_counts = np.bincount(
        train_labels,
        minlength=len(class_names)
    )


    # =========================================================
    # 6. CREATE ONE FIGURE WITH 5 SUBPLOTS
    # =========================================================
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(20, 12)
    )

    # Main title
    fig.suptitle(
        f"{model_name} — Model Diagnostics",
        fontsize=20,
        fontweight="bold"
    )


    # =========================================================
    # PLOT 1 — CONFUSION MATRIX
    # =========================================================
    plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figure=fig,
        axis=axes[0, 0]
    )

    axes[0, 0].set_title(
        "Confusion Matrix"
    )


    # =========================================================
    # PLOT 2 — F1 SCORE BY CLASS
    # =========================================================
    axes[0, 1].barh(
        class_names,
        f1_scores
    )

    axes[0, 1].set_title(
        "F1 Score by Class"
    )

    axes[0, 1].set_xlabel(
        "F1 Score"
    )

    axes[0, 1].set_xlim(
        0,
        1
    )

    axes[0, 1].grid(
        alpha=0.3
    )


    # =========================================================
    # PLOT 3 — RECALL BY CLASS
    # =========================================================
    axes[0, 2].barh(
        class_names,
        recall_scores
    )

    axes[0, 2].set_title(
        "Recall by Class"
    )

    axes[0, 2].set_xlabel(
        "Recall"
    )

    axes[0, 2].set_xlim(
        0,
        1
    )

    axes[0, 2].grid(
        alpha=0.3
    )


    # =========================================================
    # PLOT 4 — TRAINING CLASS DISTRIBUTION
    # =========================================================
    axes[1, 0].barh(
        class_names,
        train_counts
    )

    axes[1, 0].set_title(
        "Training Class Distribution"
    )

    axes[1, 0].set_xlabel(
        "Number of Images"
    )

    axes[1, 0].grid(
        alpha=0.3
    )


    # =========================================================
    # PLOT 5 — TRAINING IMAGES VS F1
    # =========================================================
    axes[1, 1].scatter(
        train_counts,
        f1_scores
    )

    for i, class_name in enumerate(class_names):

        axes[1, 1].annotate(
            class_name,
            (
                train_counts[i],
                f1_scores[i]
            )
        )

    axes[1, 1].set_title(
        "Training Images vs F1 Score"
    )

    axes[1, 1].set_xlabel(
        "Number of Training Images"
    )

    axes[1, 1].set_ylabel(
        "F1 Score"
    )

    axes[1, 1].set_ylim(
        0,
        1
    )

    axes[1, 1].grid(
        alpha=0.3
    )


    # =========================================================
    # REMOVE UNUSED 6TH SUBPLOT
    # =========================================================
    axes[1, 2].axis("off")


    plt.tight_layout()
    plt.show()


