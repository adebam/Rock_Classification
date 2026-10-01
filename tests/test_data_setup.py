# test remove_bottom_annotation
import pytest
from PIL import Image

from modular.data_setup import remove_bottom_annotation
def test_remove_bottom_annotation():
    image = Image.new(mode="RGB",size=(200, 150))
    result = remove_bottom_annotation(image=image,pixels=50)
    assert result.size == (200, 100)


# test RockClassificationDataset
from PIL import Image
import torch

from modular.data_setup import RockClassificationDataset


def test_rock_classification_dataset(tmp_path):

    # --------------------------------------------------
    # 1. Create fake class folders
    # --------------------------------------------------
    class2 = tmp_path / "class2"
    class10 = tmp_path / "class10"
    class2_13 = tmp_path / "class2_class13"

    class2.mkdir()
    class10.mkdir()
    class2_13.mkdir()

    # --------------------------------------------------
    # 2. Create fake images
    # --------------------------------------------------
    Image.new("RGB", (100, 80)).save(class2 / "rock1.jpg")
    Image.new("RGB", (100, 80)).save(class2_13 / "rock2.png")
    Image.new("RGB", (100, 80)).save(class10 / "rock3.jpg")
    # This should NOT be included
    (class2 / "notes.txt").write_text("not an image")
    # --------------------------------------------------
    # 3. Create dataset
    # --------------------------------------------------
    dataset = RockClassificationDataset(root_dir=tmp_path)
    # --------------------------------------------------
    # 4. Test class sorting
    # --------------------------------------------------
    assert dataset.classes == [
        "class2",
        "class2_class13",
        "class10"
    ]
    # --------------------------------------------------
    # 5. Test label mapping
    # --------------------------------------------------
    assert dataset.class_to_idx == {
        "class2": 0,
        "class2_class13": 1,
        "class10": 2
    }
    # --------------------------------------------------
    # 6. Test number of valid images
    # --------------------------------------------------
    assert len(dataset) == 3
    # -------------------------------------------------
    # 7. Test __getitem__
    # --------------------------------------------------
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 80, 100)
    assert label == 0
    assert dataset.classes == ["class2","class2_class13","class10"]

    # --------------------------------------------------
    # 8. Valid Extension
    # --------------------------------------------------
def test_dataset_accepts_valid_extensions(tmp_path):

    class1 = tmp_path / "class1"
    class1.mkdir()

    valid_files = ["rock1.jpg","rock2.jpeg","rock3.png","rock4.tif","rock5.tiff",]
    for filename in valid_files:Image.new("RGB", (50, 50)).save(class1 / filename)
    dataset = RockClassificationDataset(root_dir=tmp_path)
    assert len(dataset) == 5
    # --------------------------------------------------
    # 9. Invalid extension ignored
    # --------------------------------------------------
def test_dataset_ignores_invalid_extensions(tmp_path):

    class1 = tmp_path / "class1"
    class1.mkdir()
    Image.new("RGB", (50, 50)).save(class1 / "rock.jpg")
    (class1 / "notes.txt").write_text("hello")
    (class1 / "data.csv").write_text("x,y")
    dataset = RockClassificationDataset(root_dir=tmp_path)
    assert len(dataset) == 1
    # --------------------------------------------------
    # 10. Test get item
    # --------------------------------------------------

def test_dataset_getitem_returns_correct_image_and_label(tmp_path):
    class1 = tmp_path / "class1"
    class1.mkdir()
    Image.new(mode="RGB",size=(120, 80)).save(class1 / "rock.jpg")
    dataset = RockClassificationDataset(root_dir=tmp_path)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3,80,120)
    assert label == 0
    # --------------------------------------------------
    # 10. Test real data path
    # --------------------------------------------------   
def test_real_dataset_has_expected_classes():

    dataset = RockClassificationDataset(root_dir="data/carbonate_1223 _merge/PPL-1223/train")
    assert len(dataset.classes) == 18
    assert "class1" not in dataset.classes
    assert "class2_class13" in dataset.classes
    assert "class18_class19" in dataset.classes
    assert "class21_class22" in dataset.classes

# test make_stratified_subset
import torch
from torch.utils.data import TensorDataset, Subset

from modular.data_setup import make_stratified_subset

    # --------------------------------------------------
    # 1. Check check split is right 25%
    # --------------------------------------------------
def test_make_stratified_subset():
    # 100 fake samples
    X = torch.randn(100, 3)

    # 4 classes, 25 samples each
    labels = torch.tensor(
        [0] * 25 +
        [1] * 25 +
        [2] * 25 +
        [3] * 25
    )

    dataset = TensorDataset(X, labels)
    subset = make_stratified_subset(dataset=dataset,labels=labels,fraction=0.2,random_seed=42)
    # Should return a PyTorch Subset
    assert isinstance(subset, Subset)

    # 20% of 100 = 20
    assert len(subset) == 20
    
    # --------------------------------------------------
    # 2. check split is even at across classes
    # --------------------------------------------------
def test_make_stratified_subset_preserves_class_distribution():
    X = torch.randn(100, 3)
    labels = torch.tensor(
        [0] * 25 +
        [1] * 25 +
        [2] * 25 +
        [3] * 25
    )
    dataset = TensorDataset(X, labels)
    subset = make_stratified_subset(dataset=dataset,labels=labels,fraction=0.2,random_seed=42)
    subset_labels = labels[subset.indices]
    class_counts = torch.bincount(subset_labels)
    assert class_counts.tolist() == [5, 5, 5, 5]
    # --------------------------------------------------
    # 3. Check random seed will produce the same
    # --------------------------------------------------

def test_make_stratified_subset_is_reproducible():
    X = torch.randn(100, 3)
    labels = torch.tensor(
        [0] * 25 +
        [1] * 25 +
        [2] * 25 +
        [3] * 25
    )

    dataset = TensorDataset(X, labels)
    subset1 = make_stratified_subset(dataset=dataset,labels=labels,fraction=0.2,random_seed=42)
    subset2 = make_stratified_subset(dataset=dataset,labels=labels,fraction=0.2,random_seed=42)
    assert list(subset1.indices) == list(subset2.indices)

    # --------------------------------------------------
    # 3. Check different seed will produce different results
    # --------------------------------------------------
def test_make_stratified_subset_changes_with_seed():
    X = torch.randn(100, 3)
    labels = torch.tensor(
        [0] * 25 +
        [1] * 25 +
        [2] * 25 +
        [3] * 25
    )

    dataset = TensorDataset(X, labels)
    subset1 = make_stratified_subset(dataset,labels,fraction=0.2,random_seed=42)
    subset2 = make_stratified_subset(dataset,labels,fraction=0.2,random_seed=123)
    assert list(subset1.indices) != list(subset2.indices)

#subset_with_transform
import copy
import torch
from torch.utils.data import Dataset, Subset

from modular.data_setup import subset_with_transform


class DummyDataset_0(Dataset):
    def __init__(self):
        self.data = list(range(10))
        self.image_transform = "original_transform"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def test_subset_with_transform():

    # Original dataset
    dataset = DummyDataset_0()
    # Original subset
    original_indices = [1, 3, 5, 7]
    subset = Subset(dataset,original_indices)
    # New transform
    new_transform = "new_transform"
    # Run function
    new_subset = subset_with_transform(subset=subset,transform=new_transform)
    # 1. Should still be a Subset
    assert isinstance(new_subset, Subset)
    # 2. Indices should be preserved
    assert list(new_subset.indices) == original_indices
    # 3. Underlying dataset should be a DIFFERENT object
    assert new_subset.dataset is not subset.dataset
    # 4. New dataset should have the new transform
    assert new_subset.dataset.image_transform == new_transform
    # 5. Original dataset should remain unchanged
    assert subset.dataset.image_transform == "original_transform"

#merge_classes
from pathlib import Path

from modular.data_setup import merge_classes
def test_merge_classes(tmp_path):
    # --------------------------------------------------
    # 1. Build fake dataset structure
    # --------------------------------------------------
    for split in ["train", "val", "test"]:
        class2 = tmp_path / split / "class2"
        class13 = tmp_path / split / "class13"
        class2.mkdir(parents=True)
        class13.mkdir(parents=True)
        # Same filename deliberately used in both classes
        (class2 / "rock1.jpg").write_text("class2 image")
        (class13 / "rock1.jpg").write_text("class13 image")
    # --------------------------------------------------
    # 2. Run merge
    # --------------------------------------------------
    merge_classes(root_dir=tmp_path,class_a="class2",class_b="class13",new_class_name="class2_class13")
    # --------------------------------------------------
    # 3. Check every split
    # --------------------------------------------------
    for split in ["train", "val", "test"]:

        merged_dir = (
            tmp_path /
            split /
            "class2_class13"
        )

        # New merged folder should exist
        assert merged_dir.exists()

        # Old folders should be gone
        assert not (tmp_path / split / "class2").exists()
        assert not (tmp_path / split / "class13").exists()

        # Both files should exist
        assert (
            merged_dir /
            "class2__rock1.jpg"
        ).exists()

        assert (
            merged_dir /
            "class13__rock1.jpg"
        ).exists()

        # Should contain exactly 2 files
        assert len(list(merged_dir.iterdir())) == 2
        
    # --------------------------------------------------
    # 4. check that this works if merged_dir.exists():raise FileExistsError(...)
    # --------------------------------------------------
import pytest
from modular.data_setup import merge_classes
def test_merge_classes_raises_if_destination_exists(tmp_path):

    class2 = tmp_path / "train" / "class2"
    class13 = tmp_path / "train" / "class13"
    merged = tmp_path / "train" / "class2_class13"

    class2.mkdir(parents=True)
    class13.mkdir(parents=True)
    merged.mkdir(parents=True)

    with pytest.raises(FileExistsError):

        merge_classes(
            root_dir=tmp_path,
            class_a="class2",
            class_b="class13",
            new_class_name="class2_class13",
            splits=("train",)
        )
    # --------------------------------------------------
    # 4. check nested directory work
    # --------------------------------------------------
def test_merge_classes_flattens_nested_paths(tmp_path):

    nested_dir = (
        tmp_path /
        "train" /
        "class2" /
        "folder_a"
    )

    nested_dir.mkdir(parents=True)

    class13 = (
        tmp_path /
        "train" /
        "class13"
    )

    class13.mkdir(parents=True)
    (nested_dir / "rock.jpg").write_text("rock")
    (class13 / "other.jpg").write_text("rock")
    merge_classes(
        root_dir=tmp_path,
        class_a="class2",
        class_b="class13",
        new_class_name="class2_class13",
        splits=("train",)
    )
    merged_dir = (
        tmp_path /
        "train" /
        "class2_class13"
    )
    assert (
        merged_dir /
        "class2__folder_a__rock.jpg"
    ).exists()

# delete_class
from modular.data_setup import delete_class


def test_delete_class(tmp_path):

    # --------------------------------------------------
    # 1. Create fake dataset
    # --------------------------------------------------
    for split in ["train", "val", "test"]:

        class1 = tmp_path / split / "class1"
        class2 = tmp_path / split / "class2"

        class1.mkdir(parents=True)
        class2.mkdir(parents=True)

        (class1 / "rock1.jpg").write_text("rock")
        (class1 / "rock2.jpg").write_text("rock")

        (class2 / "keep_me.jpg").write_text("rock")

    # --------------------------------------------------
    # 2. Delete class1
    # --------------------------------------------------
    delete_class(
        root_dir=tmp_path,
        class_name="class1"
    )

    # --------------------------------------------------
    # 3. Verify deletion
    # --------------------------------------------------
    for split in ["train", "val", "test"]:

        assert not (
            tmp_path / split / "class1"
        ).exists()

        # Make sure another class was NOT deleted
        assert (
            tmp_path / split / "class2"
        ).exists()

        assert (
            tmp_path /
            split /
            "class2" /
            "keep_me.jpg"
        ).exists()

def test_delete_class_missing_class_does_not_fail(tmp_path):

    train_dir = tmp_path / "train"
    train_dir.mkdir()

    delete_class(
        root_dir=tmp_path,
        class_name="class1",
        splits=("train",)
    )

    assert train_dir.exists()

def test_delete_class_prints_skip_message(tmp_path, capsys):

    train_dir = tmp_path / "train"
    train_dir.mkdir()

    delete_class(
        root_dir=tmp_path,
        class_name="class1",
        splits=("train",)
    )

    captured = capsys.readouterr()

    assert "[train] class1 not found. Skipping." in captured.out


#create_transformation
import pytest
import torch

from PIL import Image
from torch.utils.data import Dataset

from modular.data_setup import create_transformation


class DummyImageDataset(Dataset):

    def __init__(self, images, image_transform=None):
        self.images = images
        self.image_transform = image_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]

        if self.image_transform:
            image = self.image_transform(image)

        return image, 0
    # --------------------------------------------------
    # 1. Test the main external-normalization behavior
    # --------------------------------------------------
def test_create_transformation_external():

    images = [
        Image.new("RGB", (64, 64))
        for _ in range(4)
    ]

    dataset = DummyImageDataset(images)

    train_transform, val_transform, mean, std, dataset_len = (
        create_transformation(
            train_dataset=dataset,
            IMG_SIZE=32,
            resize=40,
            pixels=0,
            normalization="external",
            mean_ext=[0.0, 0.0, 0.0],
            std_ext=[1.0, 1.0, 1.0],
            num_workers=0
        )
    )

    assert mean == [0.0, 0.0, 0.0]
    assert std == [1.0, 1.0, 1.0]

    assert dataset_len == 4

    image = Image.new("RGB", (64, 64))

    train_output = train_transform(image)
    val_output = val_transform(image)

    assert train_output.shape == (3, 32, 32)
    assert val_output.shape == (3, 32, 32)
    # --------------------------------------------------
    # 2. A validation must be deterministic
    # --------------------------------------------------
def test_val_transform_is_deterministic():

    images = [
        Image.new("RGB", (64, 64))
    ]

    dataset = DummyImageDataset(images)

    _, val_transform, _, _, _ = create_transformation(
        train_dataset=dataset,
        IMG_SIZE=32,
        resize=40,
        pixels=0,
        normalization="external",
        mean_ext=[0.0, 0.0, 0.0],
        std_ext=[1.0, 1.0, 1.0],
        num_workers=0
    )

    image = Image.new(
        "RGB",
        (64, 64),
        color=(100, 150, 200)
    )

    result1 = val_transform(image)
    result2 = val_transform(image)

    assert torch.equal(result1, result2)
    # --------------------------------------------------
    # 3. Test your calculated normalization
    # --------------------------------------------------
def test_create_transformation_calculates_mean_std():

    images = [
        Image.new("RGB", (64, 64), color=(0, 0, 0)),
        Image.new("RGB", (64, 64), color=(255, 255, 255)),
    ]

    dataset = DummyImageDataset(images)

    _, _, mean, std, dataset_len = create_transformation(
        train_dataset=dataset,
        IMG_SIZE=32,
        resize=40,
        pixels=0,
        normalization="mydataset",
        batch_size=2,
        num_workers=0
    )

    assert dataset_len == 2

    assert torch.allclose(
        torch.tensor(mean),
        torch.tensor([0.5, 0.5, 0.5])
    )

    assert torch.allclose(
        torch.tensor(std),
        torch.tensor([0.5, 0.5, 0.5])
    )
    # --------------------------------------------------
    # 4. Test that the original dataset is not modified
    # --------------------------------------------------

def test_create_transformation_does_not_modify_original_dataset():

    original_transform = lambda x: x

    images = [
        Image.new("RGB", (64, 64))
    ]

    dataset = DummyImageDataset(
        images,
        image_transform=original_transform
    )

    create_transformation(
        train_dataset=dataset,
        IMG_SIZE=32,
        resize=40,
        pixels=0,
        normalization="external",
        mean_ext=[0.0, 0.0, 0.0],
        std_ext=[1.0, 1.0, 1.0],
        num_workers=0
    )

    assert dataset.image_transform is original_transform

    # --------------------------------------------------
    # 5. Test your resize >= IMG_SIZE protection
    # --------------------------------------------------
def test_create_transformation_rejects_small_resize():

    dataset = DummyImageDataset([
        Image.new("RGB", (64, 64))
    ])

    with pytest.raises(AssertionError):

        create_transformation(
            train_dataset=dataset,
            IMG_SIZE=224,
            resize=200,
            pixels=0,
            normalization="external",
            mean_ext=[0.0, 0.0, 0.0],
            std_ext=[1.0, 1.0, 1.0],
            num_workers=0
        )
    # --------------------------------------------------
    # 6. Test missing external statistics
    # --------------------------------------------------
def test_external_normalization_requires_mean_and_std():

    dataset = DummyImageDataset([
        Image.new("RGB", (64, 64))
    ])

    with pytest.raises(ValueError):

        create_transformation(
            train_dataset=dataset,
            IMG_SIZE=32,
            resize=40,
            pixels=0,
            normalization="external",
            num_workers=0
        )

    # --------------------------------------------------
    # 7. Test invalid normalization names
    # --------------------------------------------------
def test_create_transformation_rejects_invalid_normalization():

    dataset = DummyImageDataset([
        Image.new("RGB", (64, 64))
    ])

    with pytest.raises(ValueError):

        create_transformation(
            train_dataset=dataset,
            IMG_SIZE=32,
            resize=40,
            pixels=0,
            normalization="wrong_option",
            num_workers=0
        )

#create_pretrained_transformation
    # --------------------------------------------------
    # 7. Test invalid normalization names
    # --------------------------------------------------
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import EfficientNet_B0_Weights

from modular.data_setup import create_pretrained_transformation


class DummyDataset(Dataset):
    def __init__(self, n=4):
        self.images = [
            Image.new("RGB", (300, 300), color=(100, 150, 200))
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], 0


def test_create_pretrained_transformation_uses_weights():

    dataset = DummyDataset(n=4)

    weights = EfficientNet_B0_Weights.DEFAULT

    train_transform, val_transform, mean, std, dataset_len = (
        create_pretrained_transformation(
            train_dataset=dataset,
            weights=weights,
            pixels=0
        )
    )

    expected_transform = weights.transforms()

    assert mean == expected_transform.mean
    assert std == expected_transform.std
    assert dataset_len == 4
    # --------------------------------------------------
    # 1. Test that the function uses the pretrained weights correctly
    # --------------------------------------------------

    
    # --------------------------------------------------
    # 2. Test the output image shape
    # --------------------------------------------------
def test_pretrained_transform_output_shape():

    dataset = DummyDataset()

    weights = EfficientNet_B0_Weights.DEFAULT

    train_transform, val_transform, _, _, _ = (
        create_pretrained_transformation(
            train_dataset=dataset,
            weights=weights,
            pixels=0
        )
    )

    image = Image.new(
        "RGB",
        (300, 300),
        color=(100, 150, 200)
    )

    train_output = train_transform(image)
    val_output = val_transform(image)

    expected_crop_size = weights.transforms().crop_size
    # A single crop dimension specifies a square image.
    expected_height = expected_crop_size[0]
    expected_width = expected_crop_size[-1]

    assert train_output.shape == (
        3,
        expected_height,
        expected_width
    )

    assert val_output.shape == (
        3,
        expected_height,
        expected_width
    )


        # --------------------------------------------------
    #3. Test that validation is deterministic
    # --------------------------------------------------
def test_pretrained_val_transform_is_deterministic():

    dataset = DummyDataset()

    weights = EfficientNet_B0_Weights.DEFAULT

    _, val_transform, _, _, _ = (
        create_pretrained_transformation(
            train_dataset=dataset,
            weights=weights,
            pixels=0
        )
    )

    image = Image.new(
        "RGB",
        (300, 300),
        color=(100, 150, 200)
    )

    output1 = val_transform(image)
    output2 = val_transform(image)

    assert torch.equal(output1, output2)


#prepare_pretrained_classification_data  
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights

from modular.data_setup import (
    RockClassificationDataset,
    prepare_pretrained_classification_data
)


def test_prepare_pretrained_classification_data(tmp_path):

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    for split_dir in [train_dir, val_dir]:

        class1 = split_dir / "class1"
        class2 = split_dir / "class2"

        class1.mkdir(parents=True)
        class2.mkdir(parents=True)

        Image.new(
            "RGB",
            (300, 300),
            color=(100, 100, 100)
        ).save(class1 / "rock1.jpg")

        Image.new(
            "RGB",
            (300, 300),
            color=(150, 150, 150)
        ).save(class2 / "rock2.jpg")

    train_dataset = RockClassificationDataset(
        root_dir=train_dir
    )

    weights = EfficientNet_B0_Weights.DEFAULT

    fig, train_loader, val_loader = (
        prepare_pretrained_classification_data(
            train_dataset=train_dataset,
            val_dir=val_dir,
            weights=weights,
            pixels=0,
            batch_size=2,
            num_workers=0,
            n=1
        )
    )

    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 2

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    assert train_images.shape[0] == 2
    assert val_images.shape[0] == 2

    assert train_images.shape[1] == 3
    assert val_images.shape[1] == 3

#prepare_classification_data
from PIL import Image
import torch

from torch.utils.data import RandomSampler, SequentialSampler

from modular.data_setup import (
    RockClassificationDataset,
    prepare_classification_data
)


def test_prepare_classification_data(tmp_path, monkeypatch):

    # --------------------------------------------------
    # 1. Create fake train/validation dataset
    # --------------------------------------------------
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    for split_dir in [train_dir, val_dir]:

        class1 = split_dir / "class1"
        class2 = split_dir / "class2"

        class1.mkdir(parents=True)
        class2.mkdir(parents=True)

        Image.new(
            "RGB",
            (64, 64),
            color=(100, 100, 100)
        ).save(class1 / "rock1.jpg")

        Image.new(
            "RGB",
            (64, 64),
            color=(150, 150, 150)
        ).save(class2 / "rock2.jpg")

    # --------------------------------------------------
    # 2. Create training dataset
    # --------------------------------------------------
    train_dataset = RockClassificationDataset(
        root_dir=train_dir
    )

    # --------------------------------------------------
    # 3. Prevent plotting during the test
    # --------------------------------------------------
    fake_fig = object()

    monkeypatch.setattr(
        "modular.data_setup.plot_transformed_images",
        lambda **kwargs: fake_fig
    )

    # --------------------------------------------------
    # 4. Run function
    # --------------------------------------------------
    fig, train_loader, val_loader = (
        prepare_classification_data(
            train_dataset=train_dataset,
            val_dir=val_dir,
            IMG_SIZE=32,
            resize=40,
            pixels=0,

            normalization="external",
            mean_ext=[0.0, 0.0, 0.0],
            std_ext=[1.0, 1.0, 1.0],

            batch_size=2,
            num_workers=0
        )
    )

    # --------------------------------------------------
    # 5. Check returned figure
    # --------------------------------------------------
    assert fig is fake_fig

    # --------------------------------------------------
    # 6. Check dataset sizes
    # --------------------------------------------------
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 2

    # --------------------------------------------------
    # 7. Check shuffling behavior
    # --------------------------------------------------
    assert isinstance(
        train_loader.sampler,
        RandomSampler
    )

    assert isinstance(
        val_loader.sampler,
        SequentialSampler
    )

    # --------------------------------------------------
    # 8. Check real batches
    # --------------------------------------------------
    train_images, train_labels = next(
        iter(train_loader)
    )

    val_images, val_labels = next(
        iter(val_loader)
    )

    assert train_images.shape == (2, 3, 32, 32)
    assert val_images.shape == (2, 3, 32, 32)

    assert train_labels.shape == (2,)
    assert val_labels.shape == (2,)
   
