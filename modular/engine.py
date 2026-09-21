from timeit import default_timer as timer
import torch
from torch import nn

from .train_test import train_test
import wandb
from modular.utility import log_confusion_matrix, log_f1_by_class, log_recall_by_class,log_training_images_vs_f1,log_confusion_matrix_with_distribution


def run_experiment(
    model_class,
    model_kwargs,
    train_dataloader,
    val_dataloader,
    train_metrics,
    test_metrics,
    device,
    class_names,
    epochs=5,

    optimizer_class=torch.optim.Adam,
    optimizer_kwargs=None,

    loss_class=nn.CrossEntropyLoss,
    loss_kwargs=None,
    run_name=None,

    seed=42
):
    """
    Creates, trains, and evaluates a PyTorch model for one experiment.
    Parameters
    ----------
    model_class : torch.nn.Module
        Model class to instantiate, e.g. TinyVGG.
    
    model_kwargs : dict
        Keyword arguments passed to the model class when creating
        a fresh model instance.
    
    train_dataloader : DataLoader
        DataLoader containing the training dataset.
    
    val_dataloader : DataLoader
        DataLoader containing the validation dataset.
    
    train_metrics :
        TorchMetrics collection used to calculate training metrics.
    
    test_metrics :
        TorchMetrics collection used to calculate validation metrics.
    
    device : torch.device
        Device used for training, e.g. "cuda" or "cpu".
    
    epochs : int
        Number of training epochs. Default is 5.
    
    optimizer_class : torch.optim.Optimizer
        PyTorch optimizer class used to train the model.
        Default is torch.optim.Adam.
    
    optimizer_kwargs : dict or None
        Keyword arguments passed to the optimizer.
        Can include values such as learning rate, weight decay,
        momentum, etc.
    
        Example:
            {
                "lr": 0.001,
                "weight_decay": 0.01
            }
    
    loss_class : torch.nn.Module
        PyTorch loss function class used to create the loss function.
        Default is nn.CrossEntropyLoss.
    
    loss_kwargs : dict or None
        Keyword arguments passed to the loss function.
    
        Can be used for options such as class weighting or
        label smoothing.
    
        Example for weighted cross entropy:
            {
                "weight": class_weights.to(device)
            }
    
        Example for label smoothing:
            {
                "label_smoothing": 0.1
            }
    
    seed : int
        Random seed used for reproducibility. Default is 42.
    
    Returns
    -------
    model : torch.nn.Module
        Trained model.
    
    results : dict
        Dictionary containing training and validation loss,
        metrics, and epoch training times.
    
    total_training_time : float
        Total training time for the experiment in seconds.

    wandb.init()
    starts a new Weights & Biases experiment run and records the main experiment configuration, such as the model, 
    number of epochs, optimizer, loss function, batch size, and model/optimizer/loss hyperparameters.
    """

    # Resolve optional settings before recording them in W&B.
    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr": 0.001}

    if loss_kwargs is None:
        loss_kwargs = {}

    # ------------------------------------------
    #0. Create W&B experiment
    # ------------------------------------------

    wandb.init(
        project="rock-classification_merged_classes",
        name=run_name,
        tags=["v1"],
        config={
            "model": model_class.__name__,
            "epochs": epochs,
            "optimizer": optimizer_class.__name__,
            "loss": loss_class.__name__,
    
            "learning_rate": optimizer_kwargs.get("lr"),
            "batch_size": train_dataloader.batch_size,
    
            **model_kwargs,
            **loss_kwargs
        }
    )
    
    # -------------------------------------------------
    # 1. Set random seeds
    # -------------------------------------------------
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # -------------------------------------------------
    # 2. Create a fresh model
    # -------------------------------------------------
    model = model_class(**model_kwargs).to(device)
    # -------------------------------------------------
    # 3. Create loss function
    # -------------------------------------------------
    loss_fn = loss_class(**loss_kwargs)
    # -------------------------------------------------
    # 4. Create optimizer
    # -------------------------------------------------
    optimizer = optimizer_class(
        model.parameters(),
        **optimizer_kwargs
    )
    # -------------------------------------------------
    # 5. Print experiment information
    # -------------------------------------------------
    print(f"Model: {model.__class__.__name__}")
    print(f"Epochs: {epochs}")
    print(f"Loss function: {loss_fn.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Optimizer settings: {optimizer_kwargs}")
    print(f"Train batch size: {train_dataloader.batch_size}")
    print(f"Validation batch size: {val_dataloader.batch_size}")
    print("-" * 50)
    # -------------------------------------------------
    # 6. Start timer
    # -------------------------------------------------
    start_time = timer()
    # -------------------------------------------------
    # 7. Train model
    # -------------------------------------------------
    results = train_test(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        device=device,
        class_names=class_names,
        epochs=epochs
    )

    # -------------------------------------------------
    # 8. Stop timer
    # -------------------------------------------------
    end_time = timer()
    total_training_time = end_time - start_time
    print("-" * 50)
    print(f"Total training time: {total_training_time:.3f} seconds")

    # Log confusion matrix to W&B
    log_confusion_matrix(model=model,dataloader=val_dataloader,class_names=class_names,device=device)
    # F1 by class
    log_f1_by_class(model=model,dataloader=val_dataloader,class_names=class_names,device=device)
    # recall by class
    log_recall_by_class(model=model,dataloader=val_dataloader,class_names=class_names,device=device)
    #training images
    log_training_images_vs_f1(model=model,train_dataloader=train_dataloader,val_dataloader=val_dataloader,class_names=class_names,device=device)
    #log_confusion_matrix_with_distribution
    log_confusion_matrix_with_distribution(model=model,dataloader=val_dataloader,class_names=class_names,device=device)

    # 9. Finish W&B experiment
    wandb.finish()

    return model, results, total_training_time
