import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modular.engine import run_experiment


# ============================================================
# Helper model
# ============================================================

class SimpleModel(nn.Module):
    def __init__(self, input_features=4, hidden_units=8, output_features=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Helper DataLoader
# ============================================================

def create_test_dataloader():

    torch.manual_seed(42)

    X = torch.randn(12, 4)
    y = torch.randint(0, 3, (12,))

    dataset = TensorDataset(X, y)

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )


# ============================================================
# Dummy metrics
# ============================================================

class DummyMetrics:

    def update(self, preds, targets):
        pass

    def compute(self):
        return {}

    def reset(self):
        pass


# ============================================================
# Test run_experiment
# ============================================================

def test_run_experiment_returns_correct_outputs(monkeypatch):

    train_dataloader = create_test_dataloader()
    val_dataloader = create_test_dataloader()

    # --------------------------------------------------------
    # Mock train_test()
    # --------------------------------------------------------

    fake_results = {
        "train_loss": [1.0],
        "train_acc": [0.5],
        "test_loss": [1.1],
        "test_acc": [0.4]
    }

    def fake_train_test(**kwargs):
        return fake_results

    monkeypatch.setattr(
        "modular.engine.train_test",
        fake_train_test
    )

    # --------------------------------------------------------
    # Mock W&B
    # --------------------------------------------------------

    monkeypatch.setattr(
        "modular.engine.wandb.init",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.wandb.finish",
        lambda *args, **kwargs: None
    )

    # --------------------------------------------------------
    # Mock utility logging functions
    # --------------------------------------------------------

    monkeypatch.setattr(
        "modular.engine.log_confusion_matrix",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_f1_by_class",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_recall_by_class",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_training_images_vs_f1",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_confusion_matrix_with_distribution",
        lambda *args, **kwargs: None
    )

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    model, results, training_time = run_experiment(
        model_class=SimpleModel,

        model_kwargs={
            "input_features": 4,
            "hidden_units": 8,
            "output_features": 3
        },

        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,

        train_metrics=DummyMetrics(),
        test_metrics=DummyMetrics(),

        device=torch.device("cpu"),

        class_names=[
            "class0",
            "class1",
            "class2"
        ],

        epochs=1,

        run_name="pytest_test"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert isinstance(model, SimpleModel)

    assert isinstance(results, dict)

    assert results == fake_results

    assert isinstance(training_time, float)

    assert training_time >= 0

def test_run_experiment_default_optimizer(monkeypatch):

    train_dataloader = create_test_dataloader()
    val_dataloader = create_test_dataloader()

    captured = {}

    def fake_train_test(**kwargs):

        captured["optimizer"] = kwargs["optimizer"]

        return {}

    monkeypatch.setattr(
        "modular.engine.train_test",
        fake_train_test
    )

    monkeypatch.setattr(
        "modular.engine.wandb.init",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.wandb.finish",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_confusion_matrix",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_f1_by_class",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_recall_by_class",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_training_images_vs_f1",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        "modular.engine.log_confusion_matrix_with_distribution",
        lambda *args, **kwargs: None
    )

    run_experiment(
        model_class=SimpleModel,

        model_kwargs={
            "input_features": 4,
            "hidden_units": 8,
            "output_features": 3
        },

        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,

        train_metrics=DummyMetrics(),
        test_metrics=DummyMetrics(),

        device=torch.device("cpu"),

        class_names=[
            "class0",
            "class1",
            "class2"
        ]
    )

    optimizer = captured["optimizer"]

    assert isinstance(
        optimizer,
        torch.optim.Adam
    )

    assert optimizer.param_groups[0]["lr"] == 0.001