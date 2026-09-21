from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
import torch
import wandb

def summarize_experiment(
    results_df,
    model,
    experiment_name,
    model_name,
    img_size,
    loss_fn,
    notes=None
):
    """
    Summarizes the final-epoch performance of a model training experiment.

    Records the training and validation metrics from the last epoch,
    along with training-time statistics and important experiment settings.

    The returned dictionary can be combined with summaries from other
    experiments to create an experiment comparison dataframe.
    """

    summary = {
        "experiment": experiment_name,
        "model": model_name,
        "img_size": img_size,
        "loss_fn": loss_fn,

        "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),

        "train_loss": results_df["train_loss"].iloc[-1].round(3),
        "train_acc": results_df["train_acc"].iloc[-1].round(3),
        "train_f1": results_df["train_f1"].iloc[-1].round(3),
        "train_precision": results_df["train_precision"].iloc[-1].round(3),
        "train_recall": results_df["train_recall"].iloc[-1].round(3),

        "val_loss": results_df["test_loss"].iloc[-1].round(3),
        "val_acc": results_df["test_acc"].iloc[-1].round(3),
        "val_f1": results_df["test_f1"].iloc[-1].round(3),
        "val_precision": results_df["test_precision"].iloc[-1].round(3),
        "val_recall": results_df["test_recall"].iloc[-1].round(3),

        "avg_epoch_time": results_df["epoch_time"].mean().round(0),
        "total_training_time": results_df["epoch_time"].sum().round(0),

        "notes": notes
    }

    return summary


#1. log_confusion_matrix
def log_confusion_matrix(
    model,
    dataloader,
    class_names,
    device,
):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    cm = wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=class_names
    )

    wandb.log({
        "Validation Confusion Matrix": cm
    })

def log_f1_by_class(
    model,
    dataloader,
    class_names,
    device,
):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    f1_metric = MulticlassF1Score(
        num_classes=len(class_names),
        average=None
    ).to(device)

    f1_scores = f1_metric(
        all_preds,
        all_labels
    ).cpu()

    table = wandb.Table(
        columns=["Class", "F1 Score"],
        data=[
            [class_name, f1.item()]
            for class_name, f1 in zip(class_names, f1_scores)
        ]
    )

    wandb.log({
        "Validation F1 by Class": wandb.plot.bar(
            table,
            "Class",
            "F1 Score",
            title="F1 Score by Class"
        )
    })

def log_recall_by_class(
    model,
    dataloader,
    class_names,
    device,
):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    recall_metric = MulticlassRecall(
        num_classes=len(class_names),
        average=None
    ).to(device)

    recall_scores = recall_metric(
        all_preds,
        all_labels
    ).cpu()

    table = wandb.Table(
        columns=["Class", "Recall"],
        data=[
            [class_name, recall.item()]
            for class_name, recall in zip(class_names, recall_scores)
        ]
    )

    wandb.log({
        "Validation Recall by Class": wandb.plot.bar(
            table,
            "Class",
            "Recall",
            title="Recall by Class"
        )
    })

def log_precision_by_class(
    model,
    dataloader,
    class_names,
    device,
):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision_metric = MulticlassPrecision(
        num_classes=len(class_names),
        average=None
    ).to(device)

    precision_scores = precision_metric(
        all_preds,
        all_labels
    ).cpu()

    table = wandb.Table(
        columns=["Class", "Precision"],
        data=[
            [class_name, precision.item()]
            for class_name, precision in zip(class_names, precision_scores)
        ]
    )

    wandb.log({
        "Validation Precision by Class": wandb.plot.bar(
            table,
            "Class",
            "Precision",
            title="Precision by Class",
        )
    })


def log_training_images_vs_f1(
    model,
    train_dataloader,
    val_dataloader,
    class_names,
    device
):
    model.eval()

    # --------------------------------------------------
    # 1. Count training images per class
    # --------------------------------------------------
    train_counts = torch.zeros(
        len(class_names),
        dtype=torch.long
    )

    for _, y in train_dataloader:
        for label in y:
            train_counts[label] += 1

    # --------------------------------------------------
    # 2. Get validation predictions
    # --------------------------------------------------
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in val_dataloader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # --------------------------------------------------
    # 3. Calculate F1 for every class
    # --------------------------------------------------
    f1_metric = MulticlassF1Score(
        num_classes=len(class_names),
        average=None
    ).to(device)

    f1_scores = f1_metric(
        all_preds,
        all_labels
    ).cpu()

    # --------------------------------------------------
    # 4. Create W&B table
    # --------------------------------------------------
    table = wandb.Table(
        columns=[
            "Class",
            "Training Images",
            "F1 Score"
        ],
        data=[
            [
                class_name,
                train_counts[i].item(),
                f1_scores[i].item()
            ]
            for i, class_name in enumerate(class_names)
        ]
    )

    # --------------------------------------------------
    # 5. Log table + scatter plot
    # --------------------------------------------------
    wandb.log({
        "training_images_vs_f1_table": table,

        "Training Images vs F1": wandb.plot.scatter(
            table,
            "Training Images",
            "F1 Score",
            title="Training Images per Class vs Validation F1"
        )
    })

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def log_confusion_matrix_with_distribution(
    model,
    dataloader,
    class_names,
    device,
    figsize=(18, 10),
    normalize=False,
):
    """
    Log a cleaner confusion matrix with validation class distribution
    aligned on the right.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    dataloader : torch.utils.data.DataLoader
        Validation dataloader.

    class_names : list[str]
        Class labels.

    device : torch.device
        Device for inference.

    figsize : tuple
        Figure size.

    normalize : bool
        If True, plot row-normalized confusion matrix.
        If False, plot raw counts.
    """

    model.eval()

    all_preds = []
    all_labels = []

    # ---------------------------------------------------------
    # 1. Collect predictions and labels
    # ---------------------------------------------------------
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    num_classes = len(class_names)

    # ---------------------------------------------------------
    # 2. Compute confusion matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=np.arange(num_classes)
    )

    # validation distribution
    split_counts = np.bincount(all_labels, minlength=num_classes)

    # keep a copy of raw counts for annotation
    cm_raw = cm.copy()

    # optional normalized version
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
    
        cm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )
    
        display_matrix = cm
        fmt_mode = "float"
    
        title = "Confusion Matrix (Row-Normalized)"
        cbar_label = "Proportion"
    
    else:
        display_matrix = cm
        fmt_mode = "int"
    
        title = "Confusion Matrix"
        cbar_label = "Count"

    # ---------------------------------------------------------
    # 3. Create figure layout
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[4.5, 1.4],
        wspace=0.15
    )

    ax_cm = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])

    # ---------------------------------------------------------
    # 4. Plot confusion matrix
    # ---------------------------------------------------------
    im = ax_cm.imshow(display_matrix, cmap="Blues")

    ax_cm.set_title(title, fontsize=15, fontweight="bold")
    ax_cm.set_xlabel("Predicted label", fontsize=12)
    ax_cm.set_ylabel("True label", fontsize=12)

    ax_cm.set_xticks(np.arange(num_classes))
    ax_cm.set_yticks(np.arange(num_classes))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_cm.set_yticklabels(class_names)

    # draw grid lines like the nice example
    ax_cm.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax_cm.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax_cm.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax_cm.tick_params(which="minor", bottom=False, left=False)

    # annotate cells
    threshold = display_matrix.max() / 2 if display_matrix.max() > 0 else 0

    for i in range(num_classes):
        for j in range(num_classes):

            if fmt_mode == "int":
                value = cm_raw[i, j]
                if value == 0:
                    text_str = ""
                else:
                    text_str = f"{value}"
                color_value = display_matrix[i, j]
            else:
                raw_value = cm_raw[i, j]
                norm_value = display_matrix[i, j]
                if raw_value == 0:
                    text_str = ""
                else:
                    text_str = f"{norm_value:.2f}\n({raw_value})"
                color_value = norm_value

            ax_cm.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color="white" if color_value > threshold else "black",
                fontsize=8
            )

    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    # ---------------------------------------------------------
    # 5. Plot validation distribution
    # ---------------------------------------------------------
    y_pos = np.arange(num_classes)

    ax_dist.barh(y_pos, split_counts)
    
    ax_dist.set_title(
        "Class Distribution",
        fontsize=13,
        fontweight="bold",
    )
    
    ax_dist.set_xlabel("Number of Images", fontsize=11)
    ax_dist.set_yticks(y_pos)
    ax_dist.set_yticklabels([])
    
    ax_dist.set_ylim(num_classes - 0.5, -0.5)
    
    ax_dist.set_xlim(
        0,
        split_counts.max() * 1.10
        if split_counts.max() > 0
        else 1
    )
    
    ax_dist.grid(axis="x", alpha=0.3)
    
    for i, count in enumerate(split_counts):
        ax_dist.text(
            count + split_counts.max() * 0.01,
            i,
            str(count),
            va="center",
            fontsize=8
        )

    fig.suptitle(
        "Confusion Matrix + Class Distribution",
        fontsize=18,
        fontweight="bold",
    )
    
    plt.tight_layout()
    
    wandb.log({
        "Confusion Matrix + Validation Distribution": wandb.Image(fig)
    })

    plt.close(fig)
    
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)