import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from modular.train_test import (
    train_step,
    test_step as evaluation_step,
    train_test,
    evaluate_model,
)


# ============================================================
# Helper functions
# ============================================================

def create_test_dataloader(
    num_samples=12,
    num_features=4,
    num_classes=3,
    batch_size=4
):
    """
    Create a small artificial classification dataset.
    """

    torch.manual_seed(42)

    X = torch.randn(num_samples, num_features)
    y = torch.randint(
        low=0,
        high=num_classes,
        size=(num_samples,)
    )

    dataset = TensorDataset(X, y)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataloader


def create_test_model(
    num_features=4,
    num_classes=3
):
    """
    Create a very small neural network for testing.
    """

    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(num_features, 8),
        nn.ReLU(),
        nn.Linear(8, num_classes)
    )

    return model


def create_train_metrics(num_classes=3):

    return MetricCollection({
        "train_accuracy": MulticlassAccuracy(
            num_classes=num_classes
        ),
        "train_f1": MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        ),
        "train_precision": MulticlassPrecision(
            num_classes=num_classes,
            average="macro"
        ),
        "train_recall": MulticlassRecall(
            num_classes=num_classes,
            average="macro"
        ),
    })


def create_test_metrics(num_classes=3):

    return MetricCollection({
        "test_accuracy": MulticlassAccuracy(
            num_classes=num_classes
        ),
        "test_f1": MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        ),
        "test_precision": MulticlassPrecision(
            num_classes=num_classes,
            average="macro"
        ),
        "test_recall": MulticlassRecall(
            num_classes=num_classes,
            average="macro"
        ),
    })


# ============================================================
# Test train_step()
# ============================================================

def test_train_step_returns_correct_outputs():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = nn.CrossEntropyLoss()
    metrics = create_train_metrics()

    train_loss, train_results = train_step(
        model=model,
        train_dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        device=device
    )

    # Check loss
    assert isinstance(train_loss, float)
    assert train_loss >= 0

    # Check expected metric keys
    assert "train_accuracy" in train_results
    assert "train_f1" in train_results
    assert "train_precision" in train_results
    assert "train_recall" in train_results

    # Check metric values are valid probabilities
    assert 0 <= train_results["train_accuracy"].item() <= 1
    assert 0 <= train_results["train_f1"].item() <= 1
    assert 0 <= train_results["train_precision"].item() <= 1
    assert 0 <= train_results["train_recall"].item() <= 1


def test_train_step_changes_model_parameters():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    loss_fn = nn.CrossEntropyLoss()
    metrics = create_train_metrics()

    # Save model parameters before training
    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    train_step(
        model=model,
        train_dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        device=device
    )

    parameters_after = list(model.parameters())

    # At least one parameter should have changed
    parameter_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after
        )
    )

    assert parameter_changed


def test_train_step_sets_model_to_training_mode():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    model.eval()

    assert model.training is False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    train_step(
        model=model,
        train_dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        metrics=create_train_metrics(),
        device=device
    )

    assert model.training is True


# ============================================================
# Test test_step()
# ============================================================

def test_test_step_returns_correct_outputs():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    loss_fn = nn.CrossEntropyLoss()
    metrics = create_test_metrics()

    test_loss, test_results = evaluation_step(
        model=model,
        test_dataloader=dataloader,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device
    )

    assert isinstance(test_loss, float)
    assert test_loss >= 0

    assert "test_accuracy" in test_results
    assert "test_f1" in test_results
    assert "test_precision" in test_results
    assert "test_recall" in test_results

    assert 0 <= test_results["test_accuracy"].item() <= 1
    assert 0 <= test_results["test_f1"].item() <= 1
    assert 0 <= test_results["test_precision"].item() <= 1
    assert 0 <= test_results["test_recall"].item() <= 1


def test_test_step_does_not_change_model_parameters():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    evaluation_step(
        model=model,
        test_dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=create_test_metrics(),
        device=device
    )

    parameters_after = list(model.parameters())

    for before, after in zip(
        parameters_before,
        parameters_after
    ):
        assert torch.equal(before, after)


def test_test_step_sets_model_to_eval_mode():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    model.train()

    assert model.training is True

    evaluation_step(
        model=model,
        test_dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=create_test_metrics(),
        device=device
    )

    assert model.training is False


# ============================================================
# Test train_test()
# ============================================================

def test_train_test_returns_correct_results(monkeypatch):

    device = torch.device("cpu")

    train_dataloader = create_test_dataloader()
    test_dataloader = create_test_dataloader()

    model = create_test_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Prevent wandb from trying to communicate with the internet
    monkeypatch.setattr(
        "modular.engine.wandb.log",
        lambda *args, **kwargs: None
    )

    epochs = 2

    results = train_test(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        train_metrics=create_train_metrics(),
        test_metrics=create_test_metrics(),
        device=device,
        class_names=["class0", "class1", "class2"],
        loss_fn=nn.CrossEntropyLoss(),
        epochs=epochs
    )

    expected_keys = [
        "train_loss",
        "train_acc",
        "train_f1",
        "train_precision",
        "train_recall",
        "test_loss",
        "test_acc",
        "test_f1",
        "test_precision",
        "test_recall",
        "epoch_time",
    ]

    for key in expected_keys:
        assert key in results

    # Every metric should contain one value per epoch
    for key in expected_keys:
        assert len(results[key]) == epochs


def test_train_test_metric_ranges(monkeypatch):

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    monkeypatch.setattr(
        "modular.engine.wandb.log",
        lambda *args, **kwargs: None
    )

    results = train_test(
        model=model,
        train_dataloader=dataloader,
        test_dataloader=dataloader,
        optimizer=optimizer,
        train_metrics=create_train_metrics(),
        test_metrics=create_test_metrics(),
        device=device,
        class_names=["class0", "class1", "class2"],
        epochs=2
    )

    metric_names = [
        "train_acc",
        "train_f1",
        "train_precision",
        "train_recall",
        "test_acc",
        "test_f1",
        "test_precision",
        "test_recall",
    ]

    for metric_name in metric_names:

        for value in results[metric_name]:

            assert 0 <= value <= 1


def test_train_test_epoch_time_positive(monkeypatch):

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    monkeypatch.setattr(
        "modular.engine.wandb.log",
        lambda *args, **kwargs: None
    )

    results = train_test(
        model=model,
        train_dataloader=dataloader,
        test_dataloader=dataloader,
        optimizer=optimizer,
        train_metrics=create_train_metrics(),
        test_metrics=create_test_metrics(),
        device=device,
        class_names=["class0", "class1", "class2"],
        epochs=2
    )

    for epoch_time in results["epoch_time"]:
        assert epoch_time >= 0


# ============================================================
# Test evaluate_model()
# ============================================================

def test_evaluate_model_returns_metrics():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    metrics = create_test_metrics()

    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        class_names=["class0", "class1", "class2"],
        device=device
    )

    assert isinstance(results, dict)

    assert "test_accuracy" in results
    assert "test_f1" in results
    assert "test_precision" in results
    assert "test_recall" in results

    assert 0 <= results["test_accuracy"].item() <= 1
    assert 0 <= results["test_f1"].item() <= 1
    assert 0 <= results["test_precision"].item() <= 1
    assert 0 <= results["test_recall"].item() <= 1


def test_evaluate_model_sets_eval_mode():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    model.train()

    assert model.training is True

    evaluate_model(
        model=model,
        dataloader=dataloader,
        metrics=create_test_metrics(),
        class_names=["class0", "class1", "class2"],
        device=device
    )

    assert model.training is False


def test_evaluate_model_does_not_change_parameters():

    device = torch.device("cpu")

    dataloader = create_test_dataloader()
    model = create_test_model()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    evaluate_model(
        model=model,
        dataloader=dataloader,
        metrics=create_test_metrics(),
        class_names=["class0", "class1", "class2"],
        device=device
    )

    parameters_after = list(model.parameters())

    for before, after in zip(
        parameters_before,
        parameters_after
    ):
        assert torch.equal(before, after)
