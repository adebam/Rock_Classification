def walk_through_dir(dir_path):
    """
    Walk through dir_path, ignoring hidden folders,
    and return a DataFrame showing the number of images
    in train, val, and test for each class.
    """
    from pathlib import Path
    import os
    import pandas as pd

    # Store counts here
    records = []

    # 1. Walk through directory and print folder contents
    for dirpath, dirnames, filenames in os.walk(dir_path):
        # Ignore hidden folders such as .ipynb_checkpoints
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        # Ignore hidden files
        filenames = [ f for f in filenames if not f.startswith(".")]
        #print(f"There are {len(dirnames)} directories " f"and {len(filenames)} images in '{dirpath}'.")

    # 2. Count images for train / val / test
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dir_path, split)
        # Skip split if it doesn't exist
        if not os.path.isdir(split_path):
            continue
        # Each folder inside split_path is a class
        for class_name in os.listdir(split_path):
            # Ignore hidden folders
            if class_name.startswith("."):
                continue
            class_path = os.path.join(split_path, class_name)
            # Make sure it is actually a directory
            if not os.path.isdir(class_path):
                continue
            # Count files inside the class folder
            num_images = len([
                file
                for file in os.listdir(class_path)
                if not file.startswith(".")
                and os.path.isfile(os.path.join(class_path, file))
            ])
            records.append({
                "class": class_name,
                "split": split,
                "num_images": num_images
            })

    # 3. Convert to DataFrame
    counts_df = pd.DataFrame(records)

    counts_df = (
        counts_df
        .pivot(index="class", columns="split", values="num_images")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    # Make sure columns exist even if a split is missing
    for split in ["train", "val", "test"]:
        if split not in counts_df.columns:
            counts_df[split] = 0

    # Put columns in desired order
    counts_df = counts_df[
        ["class", "train", "val", "test"]
    ]
    # Add total images for each class
    counts_df["total"] = (
        counts_df["train"]
        + counts_df["val"]
        + counts_df["test"]
    )
   # Percentage of entire dataset represented by each class
    counts_df["percent"] = (counts_df["total"] / counts_df["total"].sum() * 100).round(2)

    # Sort classes numerically: class1, class2, ..., class10, class11, ...
    counts_df["class_number"] = (counts_df["class"].str.extract(r"(\d+)").astype(int))

    counts_df = (counts_df.sort_values("class_number").drop(columns="class_number").reset_index(drop=True))
    return counts_df

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RockClassificationDataset(Dataset):

    def __init__(self, root_dir, image_transform=None):
        """
        Args:
            root_dir:
                Path to one split, for example:
                data/carbonate_1223/PPL-1223/train

                Expected structure:

                train/
                    class_1/
                        image1.png
                        image2.png
                    class_2/
                        image3.png
                        image4.png

            image_transform:
                Optional transformations applied to the images.
        """

        self.root_dir = Path(root_dir)
        self.image_transform = image_transform

        # 1. Find all class folders
        def class_sort_key(name):
            return tuple(
                int(part)
                for part in name.removeprefix("class").split("_class")
            )
        
        self.classes = sorted(
            [
                folder.name
                for folder in self.root_dir.iterdir()
                if folder.is_dir()
            ],
            key=class_sort_key
        )
        # 2. Convert class names into numeric labels
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # 3. Collect image paths and their labels
        self.samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        for class_name in self.classes:
            class_folder = self.root_dir / class_name
            label = self.class_to_idx[class_name]
            for image_path in sorted(class_folder.iterdir()):
                if image_path.suffix.lower() in valid_extensions:
                    self.samples.append(
                        (image_path, label)
                    )
    def __len__(self):
        """
        Return number of images.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one rock image and its class label.
        """
        # 1. Get image path and label
        image_path, label = self.samples[idx]
        # 2. Load image
        image = Image.open(image_path).convert("RGB")
        # 3. Apply transformations
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = transforms.ToTensor()(image)
        # 4. Return X and y
        return image, label

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
import torch

def make_stratified_subset(dataset: torch.utils.data.Dataset, labels: list, fraction: float=0.2, random_seed=42):
    """
    Create a smaller stratified subset of a PyTorch Dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Original training dataset.

    labels : list or array
        Class label for every image in the dataset.

    fraction : float
        Fraction of dataset to keep.
        Example: 0.2 = keep 20%.

    random_seed : int
        Reproducibility seed.

    Returns
    -------
    torch.utils.data.Subset
    """

    indices = np.arange(len(dataset))

    subset_indices, _ = train_test_split(indices,train_size=fraction,stratify=labels,random_state=random_seed)

    return Subset(dataset, subset_indices)
import torch
def calculate_mean_std(dataloader):
    """
    Calculate the RGB mean and standard deviation using float64.
    """
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_squared = torch.zeros(3, dtype=torch.float64)
    num_pixels = 0

    for images, _ in dataloader:
        images = images.to(dtype=torch.float64)

        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_squared += images.square().sum(dim=(0, 2, 3))

        num_pixels += (
            images.size(0)
            * images.size(2)
            * images.size(3)
        )

    if num_pixels == 0:
        raise ValueError("Cannot calculate statistics from an empty dataset.")

    mean = channel_sum / num_pixels

    variance = (
        channel_sum_squared / num_pixels
        - mean.square()
    ).clamp_min(0)

    std = torch.sqrt(variance)

    return mean, std

from torchvision import transforms
def remove_bottom_annotation(image, pixels):
    """
    this function can be used to remove the bottom part. It removes the ruler.
    """
    width, height = image.size
    if not 0 <= pixels < height:
        raise ValueError(f"pixels must be between 0 and {height - 1}, got {pixels}")
    return image.crop((0, 0, width, height - pixels))
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset, Subset
import copy

def create_transformation(
    train_dataset: Dataset, 
    IMG_SIZE:int =224,
    resize:int =256,
    pixels: int = 90,
    normalization: str = "mydataset",
    mean_ext=None,
    std_ext=None,

    batch_size:int =32,
    num_workers:int = 4,
    #RandomResizedCrop_scale_min: float =0.8,
    #RandomResizedCrop_scale_max: float =1.0,
    #RandomResizedCrop_ratio_min: float =0.9,
    #RandomResizedCrop_ratio_max: float =1.1,
    RandomHorizontalFlip_p: float = 0.5,
    RandomVerticalFlip_p:float = 0.0,  # never flip for now. #If bedding direction, sedimentary structures, laminations, or other directional geological features matter, test whether vertical flip helps or hurts.
    #RandomRotation_degrees:int = 0,  # keep this as zeros as it creates issues
    brightness:float = 0.15,
    contrast:float = 0.15,
    saturation:float = 0.1,
    hue:float = 0.03,
):
    """
Creates training and validation/test image transformations for a rock classification dataset.

Args:
    train_dataset: A PyTorch Dataset or Subset used to calculate normalization
        statistics and determine the training dataset size.
    IMG_SIZE: Final image size passed to the model. Default is 224.
    resize: Intermediate resize value used before CenterCrop for statistics
        calculation and validation/test preprocessing. Default is 256.
    normalization: Normalization method to use. Should be either:
        "mydataset" to calculate mean and standard deviation from train_dataset,
        or "external" to use externally supplied mean and standard deviation values.
    mean_ext: External RGB mean values used when normalization="external".
    std_ext: External RGB standard deviation values used when normalization="external".
    batch_size: Batch size used when calculating dataset mean and standard deviation.
    num_workers: Number of DataLoader worker processes used when calculating
        dataset statistics.
    RandomResizedCrop_scale_min: Minimum crop area scale for RandomResizedCrop.
    RandomResizedCrop_scale_max: Maximum crop area scale for RandomResizedCrop.
    RandomResizedCrop_ratio_min: Minimum aspect ratio for RandomResizedCrop.
    RandomResizedCrop_ratio_max: Maximum aspect ratio for RandomResizedCrop.
    RandomHorizontalFlip_p: Probability of applying a horizontal flip.
    RandomVerticalFlip_p: Probability of applying a vertical flip.
    RandomRotation_degrees: Maximum rotation angle in degrees. Images may be
        rotated between -degrees and +degrees.
    brightness: Maximum brightness variation used by ColorJitter.
    contrast: Maximum contrast variation used by ColorJitter.
    saturation: Maximum saturation variation used by ColorJitter.
    hue: Maximum hue variation used by ColorJitter.
    pixels: Number of pixels removed from the bottom of each image
    to remove the scale bar/annotation. Default is 80.

Returns:
    train_transform: PyTorch transformation pipeline containing training
        augmentations and normalization.
    val_test_transform: Deterministic transformation pipeline for validation
        and test images.
    mean: RGB mean values used for normalization.
    std: RGB standard deviation values used for normalization.
    dataset_len: Number of samples in the supplied training dataset.

Example usage:
    train_transform, val_test_transform, mean, std, dataset_len = create_transformation(
        train_dataset=train_subset_20_dataset,
        IMG_SIZE=224,
        resize=256,
        normalization="mydataset"
    )

    # Using ImageNet statistics for a pretrained model:
    train_transform, val_test_transform, mean, std, dataset_len = create_transformation(
        train_dataset=train_subset_20_dataset,
        normalization="external",
        mean_ext=[0.485, 0.456, 0.406],
        std_ext=[0.229, 0.224, 0.225]
    )
"""
    # 1. Write separate transforms for train and test data
    from torchvision import transforms
    from torch.utils.data import DataLoader

    #IMG_SIZE = IMG_SIZE # can change this for different experiments
    # ============================================================
    # 1. CALCULATE TRAINING DATASET STATISTICS
    # ============================================================
    #------------------------------------------------------------------------------------------
    # Write a transform to create my own statistics for the standardization
    assert resize >= IMG_SIZE, "resize must be greater than or equal to IMG_SIZE"
    stats_transform = transforms.Compose([
        transforms.Lambda(lambda img: remove_bottom_annotation(img, pixels=pixels)),
        transforms.Resize(resize),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor()
    ])

    if isinstance(train_dataset, Subset):
        # Copy the original dataset underlying the subset
        train_dataset_stats_full = copy.copy(train_dataset.dataset)
        # Give the copy the deterministic statistics transform
        train_dataset_stats_full.image_transform = stats_transform
        # Keep exactly the same subset indices
        train_dataset_stats = Subset(train_dataset_stats_full,train_dataset.indices)
    else:
        # Copy the full dataset
        train_dataset_stats = copy.copy(train_dataset)

        # Change only the transform on the copy
        train_dataset_stats.image_transform = stats_transform

    # Make dataloader for stats if necessary  Then calculate the mean and standard deviation. Have the option to ue my dataset, or an external one. 
    if normalization == "mydataset":
        # Calculate statistics from your selected training dataset
        stats_loader = DataLoader(train_dataset_stats,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        mean, std = calculate_mean_std(stats_loader)
        mean = mean.tolist()
        std = std.tolist()
    elif normalization == "external":

        if mean_ext is None or std_ext is None:
            raise ValueError("mean_ext and std_ext must be provided when normalization='external'")
        mean = mean_ext
        std = std_ext
    else:
        raise ValueError("normalization must be either 'mydataset' or 'external'")

    #######################################
    #2. TRAINING TRANSFORM
    ########################################
    #------------------------------------------------------------------------------------------
    train_transform = transforms.Compose([  
        transforms.Lambda(lambda img: remove_bottom_annotation(img, pixels=pixels)),
        #transforms.RandomRotation(degrees=RandomRotation_degrees),# Small rotation
        #transforms.RandomResizedCrop(IMG_SIZE, scale=(RandomResizedCrop_scale_min, RandomResizedCrop_scale_max),ratio=(RandomResizedCrop_ratio_min, RandomResizedCrop_ratio_max)),
        transforms.Resize(resize),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=RandomHorizontalFlip_p),# Orientation augmentation
        transforms.RandomVerticalFlip(p=RandomVerticalFlip_p), #If bedding direction, sedimentary structures, laminations, or other directional geological features matter, test whether vertical flip helps or hurts.
        transforms.ColorJitter(brightness=brightness,contrast=contrast,saturation=saturation,hue=hue),# Lighting/camera variation
        #transforms.TrivialAugmentWide(num_magnitude_bins=31), can experiment with this later
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
     ]) 
    dataset_len = len(train_dataset)
    # ============================================================
    # 3. VALIDATION / TEST TRANSFORM
    # ============================================================
    val_test_transform = transforms.Compose([
        transforms.Lambda(lambda img: remove_bottom_annotation(img, pixels=pixels)),
        transforms.Resize(resize),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    return train_transform, val_test_transform, mean, std,dataset_len
import copy
from torch.utils.data import Subset

def subset_with_transform(subset, transform):
    """
    Create an independent copy of a Subset's underlying dataset
    and attach a new transform while preserving the same subset indices.
    """

    dataset_copy = copy.copy(subset.dataset)
    dataset_copy.image_transform = transform

    return Subset(
        dataset_copy,
        subset.indices
    )

from torch.utils.data import DataLoader, Subset
from modular.visualization import plot_transformed_images

def prepare_classification_data(
    train_dataset,
    val_dir,

    # Image preprocessing
    IMG_SIZE: int = 224,
    resize: int = 256,
    pixels: int = 90,

    # Normalization
    normalization: str = "mydataset",
    mean_ext=None,
    std_ext=None,

    # Augmentation
    RandomHorizontalFlip_p: float = 0.5,
    RandomVerticalFlip_p: float = 0.0,
    brightness: float = 0.15,
    contrast: float = 0.15,
    saturation: float = 0.1,
    hue: float = 0.03,

    # DataLoader
    batch_size: int = 32,
    num_workers: int = 4,

    # Plotting
    n: int = 5,
    seed: int = 42
):
    """
    Prepare training and validation data for a rock-classification experiment.

    This function:

    1. Creates training and validation/test transformations.
    2. Calculates mean and standard deviation from the training dataset.
    3. Plots randomly selected transformed training images.
    4. Applies the training transformation to the training subset.
    5. Creates the validation dataset using the validation transformation.
    6. Creates training and validation DataLoaders.

    Parameters
    ----------
    train_dataset : Dataset or Subset
        Training dataset used for the experiment.

    val_dir : str or Path
        Path to the validation dataset directory.

    IMG_SIZE : int
        Final image size passed to the model.

    resize : int
        Intermediate resize before cropping.

    pixels : int
        Number of pixels removed from the bottom of each image.

    normalization : str
        "mydataset" or "external".

    mean_ext : list, optional
        External normalization mean.

    std_ext : list, optional
        External normalization standard deviation.

    RandomHorizontalFlip_p : float
        Probability of horizontal flipping.

    RandomVerticalFlip_p : float
        Probability of vertical flipping.

    brightness : float
        ColorJitter brightness.

    contrast : float
        ColorJitter contrast.

    saturation : float
        ColorJitter saturation.

    hue : float
        ColorJitter hue.

    batch_size : int
        DataLoader batch size.

    num_workers : int
        Number of DataLoader workers.

    n : int
        Number of transformed images to display.

    seed : int
        Random seed used for plotting.

    Returns
    -------
    fig
        Figure containing examples of transformed training images.

    train_dataloader
        Training DataLoader.

    val_dataloader
        Validation DataLoader.
    """

    # ---------------------------------------------------------
    # 1. Create transformations
    # ---------------------------------------------------------
    train_transform, val_test_transform, mean, std, dataset_len = (
        create_transformation(
            train_dataset=train_dataset,
            IMG_SIZE=IMG_SIZE,
            resize=resize,
            pixels=pixels,
            normalization=normalization,
            mean_ext=mean_ext,
            std_ext=std_ext,
            batch_size=batch_size,
            num_workers=num_workers,
            RandomHorizontalFlip_p=RandomHorizontalFlip_p,
            RandomVerticalFlip_p=RandomVerticalFlip_p,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    )

    # ---------------------------------------------------------
    # 2. Automatically get image paths from training dataset
    # ---------------------------------------------------------
    if isinstance(train_dataset, Subset):

        image_paths = [
            train_dataset.dataset.samples[i][0]
            for i in train_dataset.indices
        ]

    else:

        image_paths = [
            image_path
            for image_path, _ in train_dataset.samples
        ]

    # ---------------------------------------------------------
    # 3. Plot transformed images
    # ---------------------------------------------------------
    fig = plot_transformed_images(
        image_paths=image_paths,
        transform=train_transform,
        mean=mean,
        std=std,
        n=n,
        seed=seed
    )

    # ---------------------------------------------------------
    # 4. Apply transformation to training dataset
    # ---------------------------------------------------------
    if isinstance(train_dataset, Subset):

        train_dataset_transformed = subset_with_transform(
            train_dataset,
            train_transform
        )

    else:

        train_dataset_transformed = copy.copy(train_dataset)
        train_dataset_transformed.image_transform = train_transform

    # ---------------------------------------------------------
    # 5. Create training DataLoader
    # ---------------------------------------------------------
    train_dataloader = DataLoader(
        train_dataset_transformed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # ---------------------------------------------------------
    # 6. Create validation dataset
    # ---------------------------------------------------------
    val_dataset = RockClassificationDataset(
        root_dir=val_dir,
        image_transform=val_test_transform
    )

    # ---------------------------------------------------------
    # 7. Create validation DataLoader
    # ---------------------------------------------------------
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # ---------------------------------------------------------
    # 8. Print experiment information
    # ---------------------------------------------------------
    print(f"Training images: {dataset_len}")
    print(f"Image size:      {IMG_SIZE} x {IMG_SIZE}")
    print(f"Batch size:      {batch_size}")
    print(f"Train batches:   {len(train_dataloader)}")
    print(f"Val batches:     {len(val_dataloader)}")
    print(f"Mean:            {mean}")
    print(f"Std:             {std}")

    return fig, train_dataloader, val_dataloader


    #----------------------------------------------------------------------------------------------------------------------------------------------------#

def create_pretrained_transformation(
    train_dataset: Dataset,
    weights,
    pixels: int = 90,

    RandomHorizontalFlip_p: float = 0.5,
    RandomVerticalFlip_p: float = 0.0,

    brightness: float = 0.15,
    contrast: float = 0.15,
    saturation: float = 0.1,
    hue: float = 0.03,
):
    """
    Creates training and validation/test transformations for a pretrained
    torchvision image-classification model.

    The preprocessing requirements are automatically obtained from the
    supplied pretrained weights.

    Parameters
    ----------
    train_dataset : Dataset or Subset
        Training dataset. Used to determine the number of training samples.

    weights
        TorchVision pretrained weights, for example:
        torchvision.models.EfficientNet_B0_Weights.DEFAULT

    pixels : int
        Number of pixels removed from the bottom of each image to remove
        the scale bar/annotation.

    RandomHorizontalFlip_p : float
        Probability of horizontal flipping during training.

    RandomVerticalFlip_p : float
        Probability of vertical flipping during training.

    brightness : float
        Brightness variation for ColorJitter.

    contrast : float
        Contrast variation for ColorJitter.

    saturation : float
        Saturation variation for ColorJitter.

    hue : float
        Hue variation for ColorJitter.

    Returns
    -------
    train_transform
        Training transformation including augmentation and pretrained
        normalization.

    val_test_transform
        Deterministic validation/test transformation matching the
        pretrained weights.

    mean
        RGB mean expected by the pretrained weights.

    std
        RGB standard deviation expected by the pretrained weights.

    dataset_len
        Number of samples in the training dataset.
    """

    from torchvision import transforms

    # ============================================================
    # 1. GET PREPROCESSING REQUIREMENTS FROM PRETRAINED WEIGHTS
    # ============================================================
    pretrained_transform = weights.transforms()
    crop_size = pretrained_transform.crop_size
    resize_size = pretrained_transform.resize_size
    mean = pretrained_transform.mean
    std = pretrained_transform.std
    interpolation = pretrained_transform.interpolation
    antialias = pretrained_transform.antialias

    # ============================================================
    # 2. TRAINING TRANSFORM
    # ============================================================
    train_transform = transforms.Compose([
        # Remove scale bar / annotation
        transforms.Lambda(lambda img: remove_bottom_annotation(img, pixels=pixels)),
        
        # Use resize settings required by pretrained weights
        transforms.Resize(resize_size,interpolation=interpolation,antialias=antialias),

        # Random crop instead of center crop for training augmentation
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(p=RandomHorizontalFlip_p),
        transforms.RandomVerticalFlip(p=RandomVerticalFlip_p),
        transforms.ColorJitter(brightness=brightness,contrast=contrast,saturation=saturation,hue=hue),
        transforms.ToTensor(),

        # Use normalization expected by pretrained weights
        transforms.Normalize(mean=mean,std=std)
    ])

    # ============================================================
    # 3. VALIDATION / TEST TRANSFORM
    # ============================================================
    val_test_transform = transforms.Compose([
        # Remove scale bar / annotation first
        transforms.Lambda(lambda img: remove_bottom_annotation(img, pixels=pixels)),

        # Then use EXACT preprocessing supplied by pretrained weights
        pretrained_transform
    ])

    # ============================================================
    # 4. DATASET INFORMATION
    # ============================================================
    dataset_len = len(train_dataset)
    # ============================================================
    # 5. PRINT PREPROCESSING INFORMATION
    # ============================================================
    print(f"Pretrained weights: {weights}")
    print(f"Training images:    {dataset_len}")
    print(f"Resize size:        {resize_size}")
    print(f"Crop size:          {crop_size}")
    print(f"Mean:               {mean}")
    print(f"Std:                {std}")
    print(f"Interpolation:      {interpolation}")
    print(f"Antialias:          {antialias}")

    return (train_transform,val_test_transform,mean,std,dataset_len)

def prepare_pretrained_classification_data(
    train_dataset,
    val_dir,
    weights,

    # Image preprocessing
    pixels: int = 90,

    # Augmentation
    RandomHorizontalFlip_p: float = 0.5,
    RandomVerticalFlip_p: float = 0.0,
    brightness: float = 0.15,
    contrast: float = 0.15,
    saturation: float = 0.1,
    hue: float = 0.03,

    # DataLoader
    batch_size: int = 32,
    num_workers: int = 4,

    # Plotting
    n: int = 5,
    seed: int = 42
):
    """
    Prepare training and validation data for a pretrained
    rock-classification experiment.

    The preprocessing requirements such as image size, resize size,
    normalization mean/std, interpolation, and antialiasing are obtained
    automatically from the supplied torchvision pretrained weights.

    This function:

    1. Creates pretrained training and validation/test transformations.
    2. Obtains normalization statistics from the pretrained weights.
    3. Plots randomly selected transformed training images.
    4. Applies the training transformation to the training dataset/subset.
    5. Creates the validation dataset using the pretrained validation transform.
    6. Creates training and validation DataLoaders.
    7. Prints preprocessing and DataLoader information.

    Parameters
    ----------
    train_dataset : Dataset or Subset
        Training dataset used for the experiment.

    val_dir : str or Path
        Path to the validation dataset directory.

    weights
        TorchVision pretrained weights.

        Example:
        torchvision.models.EfficientNet_B0_Weights.DEFAULT

    pixels : int
        Number of pixels removed from the bottom of each image.

    RandomHorizontalFlip_p : float
        Probability of horizontal flipping.

    RandomVerticalFlip_p : float
        Probability of vertical flipping.

    brightness : float
        ColorJitter brightness.

    contrast : float
        ColorJitter contrast.

    saturation : float
        ColorJitter saturation.

    hue : float
        ColorJitter hue.

    batch_size : int
        DataLoader batch size.

    num_workers : int
        Number of DataLoader workers.

    n : int
        Number of transformed images to display.

    seed : int
        Random seed used for plotting.

    Returns
    -------
    fig
        Figure containing examples of transformed training images.

    train_dataloader
        Training DataLoader.

    val_dataloader
        Validation DataLoader.
    """

    # ---------------------------------------------------------
    # 1. Create pretrained transformations
    # ---------------------------------------------------------
    train_transform, val_test_transform, mean, std, dataset_len = (
        create_pretrained_transformation(
            train_dataset=train_dataset,
            weights=weights,
            pixels=pixels,

            RandomHorizontalFlip_p=RandomHorizontalFlip_p,
            RandomVerticalFlip_p=RandomVerticalFlip_p,

            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    )

    # ---------------------------------------------------------
    # 2. Get preprocessing information from pretrained weights
    # ---------------------------------------------------------
    pretrained_transform = weights.transforms()

    crop_size = pretrained_transform.crop_size
    resize_size = pretrained_transform.resize_size
    interpolation = pretrained_transform.interpolation
    antialias = pretrained_transform.antialias

    # ---------------------------------------------------------
    # 3. Automatically get image paths from training dataset
    # ---------------------------------------------------------
    if isinstance(train_dataset, Subset):
        image_paths = [train_dataset.dataset.samples[i][0] for i in train_dataset.indices]
    else:
        image_paths = [image_path for image_path, _ in train_dataset.samples]
    # ---------------------------------------------------------
    # 4. Plot transformed training images
    # ---------------------------------------------------------
    fig = plot_transformed_images(image_paths=image_paths,transform=train_transform,mean=mean,std=std,n=n,seed=seed)
    # ---------------------------------------------------------
    # 5. Apply transformation to training dataset
    # ---------------------------------------------------------
    if isinstance(train_dataset, Subset):
        train_dataset_transformed = subset_with_transform(train_dataset,train_transform)
    else:
        train_dataset_transformed = copy.copy(train_dataset)
        train_dataset_transformed.image_transform = train_transform
    # ---------------------------------------------------------
    # 6. Create training DataLoader
    # ---------------------------------------------------------
    train_dataloader = DataLoader(train_dataset_transformed,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    # ---------------------------------------------------------
    # 7. Create validation dataset
    # ---------------------------------------------------------
    val_dataset = RockClassificationDataset(root_dir=val_dir,image_transform=val_test_transform)

    # ---------------------------------------------------------
    # 8. Create validation DataLoader
    # ---------------------------------------------------------
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    # ---------------------------------------------------------
    # 9. Print experiment information
    # ---------------------------------------------------------
    print("\nPRETRAINED DATA PREPARATION")
    print("-" * 50)

    print(f"Weights:          {weights}")
    print(f"Training images:  {dataset_len}")
    print(f"Resize size:      {resize_size}")
    print(f"Crop size:        {crop_size}")
    print(f"Batch size:       {batch_size}")
    print(f"Train batches:    {len(train_dataloader)}")
    print(f"Val batches:      {len(val_dataloader)}")
    print(f"Mean:             {mean}")
    print(f"Std:              {std}")
    print(f"Interpolation:    {interpolation}")
    print(f"Antialias:        {antialias}")

    print("-" * 50)

    return fig, train_dataloader, val_dataloader

def create_dinov2_transformation(
    train=True,
    IMG_SIZE=224,
    resize=256,
    RandomHorizontalFlip_p=0.5
):
    if train:
        transform = transforms.Compose([
            transforms.Lambda(
                lambda img: remove_bottom_annotation(img, pixels=90)
            ),

            transforms.RandomResizedCrop(
                IMG_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),

            transforms.RandomHorizontalFlip(
                p=RandomHorizontalFlip_p
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:
        transform = transforms.Compose([
            transforms.Lambda(
                lambda img: remove_bottom_annotation(img, pixels=90)
            ),

            transforms.Resize(
                resize,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),

            transforms.CenterCrop(IMG_SIZE),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transform


def get_dataset_labels(dataset):
    """
    Extract labels without assuming a specific dataset implementation.

    Handles:
    - torch.utils.data.Subset
    - datasets with .targets
    - datasets with .labels
    - datasets with .samples
    - fallback to dataset[i][1]
    """

    if isinstance(dataset, Subset):

        parent_labels = np.asarray(
            get_dataset_labels(dataset.dataset)
        )

        return parent_labels[
            np.asarray(dataset.indices)
        ]

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)

    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)

    if hasattr(dataset, "samples"):
        return np.asarray([
            sample[1]
            for sample in dataset.samples
        ])

    # Fallback
    return np.asarray([
        dataset[i][1]
        for i in range(len(dataset))
    ])

from pathlib import Path
import shutil


def delete_class(
    root_dir,
    class_name,
    splits=("train", "val", "test")
):
    """
    Permanently delete a class folder from train/val/test.

    Example:
        delete_class(
            root_dir="/path/to/DeepCarbonate_merged",
            class_name="class1"
        )
    """

    root_dir = Path(root_dir)

    for split in splits:
        class_dir = root_dir / split / class_name

        if class_dir.exists():
            num_files = sum(
                1 for path in class_dir.rglob("*")
                if path.is_file()
            )

            shutil.rmtree(class_dir)

            print(
                f"[{split}] Deleted {class_name} "
                f"({num_files} files)"
            )

        else:
            print(
                f"[{split}] {class_name} not found. Skipping."
            )
def merge_classes(
    root_dir,
    class_a,
    class_b,
    new_class_name,
    splits=("train", "val", "test")
):
    """
    Permanently merge two class folders into a new class folder.

    Files are renamed with their original class name as a prefix
    so duplicate filenames cannot overwrite each other.

    Example:
        class2/image001.jpg
        class13/image001.jpg

    becomes:

        class2_class13/
            class2__image001.jpg
            class13__image001.jpg
    """

    root_dir = Path(root_dir)

    for split in splits:

        split_dir = root_dir / split

        class_a_dir = split_dir / class_a
        class_b_dir = split_dir / class_b
        merged_dir = split_dir / new_class_name

        if not class_a_dir.exists():
            print(
                f"[{split}] {class_a} not found. "
                "Skipping this split."
            )
            continue

        if not class_b_dir.exists():
            print(
                f"[{split}] {class_b} not found. "
                "Skipping this split."
            )
            continue

        # Prevent accidentally merging into an existing class
        if merged_dir.exists():
            raise FileExistsError(
                f"{merged_dir} already exists."
            )

        merged_dir.mkdir(parents=True)

        total_moved = 0

        for source_dir, source_class in [
            (class_a_dir, class_a),
            (class_b_dir, class_b)
        ]:

            for file_path in source_dir.rglob("*"):

                if not file_path.is_file():
                    continue

                relative_path = file_path.relative_to(source_dir)

                # Convert any nested path to a safe filename
                relative_name = "__".join(relative_path.parts)

                new_filename = (
                    f"{source_class}__{relative_name}"
                )

                destination = merged_dir / new_filename

                shutil.move(
                    str(file_path),
                    str(destination)
                )

                total_moved += 1

        # Delete the now-empty original folders
        shutil.rmtree(class_a_dir)
        shutil.rmtree(class_b_dir)

        print(
            f"[{split}] Merged {class_a} + {class_b} "
            f"→ {new_class_name} "
            f"({total_moved} files)"
        )