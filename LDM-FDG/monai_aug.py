import numpy as np
import random
from monai.transforms import (
    Compose,
    LoadImaged,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandAffined,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    RandGaussianSmoothd,
    RandBiasFieldd,
    RandCoarseDropoutd,
    RandGridDistortiond, # This is the transform we will use for non-linear deformation
)
from monai.data import decollate_batch
import torch
from pathlib import Path
import nibabel as nib

class MonaiAugmentations2D:
    def __init__(self, keys=("image",)):
        self.keys = keys
        self.rescale = ScaleIntensityRanged(
            keys=keys,
            a_min=0.0,
            a_max=1.0,
            b_min=-1.0,
            b_max=1.0,
            clip=True,
        )

        self.ensure_channel_first = EnsureChannelFirstd(keys=keys)

        self.affine_transform = RandAffined(
            keys=keys,
            prob=1.0,
            spatial_size=None,
            rotate_range=(np.pi/36,),
            scale_range=(0.1, 0.1),
            translate_range=(0.1, 0.1),
            padding_mode="border",
            mode="bilinear",
        )
        
        # RandGridDistortiond will be instantiated and applied *within* __call__
        # to allow its parameters to randomize per subject, mirroring original TorchIO 'elastix'.


    def __call__(self, data):
        subject = data.copy()

        subject = self.ensure_channel_first(subject)

        # Apply mandatory affine transform
        subject = self.affine_transform(subject)

        # --- MANDATORY NON-LINEAR DEFORMATION using RandGridDistortiond ---
        # This replaces the always-applied elastic/coarse field transform from TorchIO.
        # Parameters are randomized per call, mimicking original 'elastix' behavior.
        num_cells_h = random.randint(2, 5) # Example: 2x2 to 5x5 grid for deformation control points
        num_cells_w = random.randint(2, 5)
        max_distort = random.uniform(2, 10) # Max displacement for control points in pixels/voxels
        
        mandatory_grid_distortion = RandGridDistortiond(
            keys=self.keys,
            prob=1.0, # Always applies when this transform object is called
            num_cells=(num_cells_h, num_cells_w),
            distort_limit=(max_distort, max_distort),
            mode="bilinear",
            padding_mode="border",
        )
        subject = mandatory_grid_distortion(subject)
        # --- END MANDATORY NON-LINEAR DEFORMATION ---

        aug_level = random.randint(1, 3)

        # Define individual transformation functions (RandGridDistortiond is no longer here)
        def monai_blur():
            downsampling_factor = random.randint(2, 3)
            std_val = float(downsampling_factor) / 2.0
            return RandGaussianSmoothd(keys=self.keys, sigma_x=(std_val, std_val),sigma_y=(std_val, std_val), prob=1.0)

        def monai_anisotropy():
            return RandCoarseDropoutd(
                keys=self.keys,
                prob=1.0,
                dropout_rate=random.uniform(0.05, 0.2),
                spatial_size=random.choice([2, 4]),
            )

        def monai_noise():
            std_range = (0.0, 0.25)
            return RandGaussianNoised(keys=self.keys, prob=1.0, std=std_range)

        def monai_field_bias():
            return RandBiasFieldd(
                keys=self.keys,
                prob=1.0,
                coeff_range=(0.0, 0.5),
                degree=3,
            )

        # List of functions (RandGridDistortiond is excluded as it's now always applied)
        all_monai_functions = [
            monai_blur,
            monai_anisotropy,
            monai_noise,
            monai_field_bias,
        ]
        blur_monai_functions = [monai_noise, monai_field_bias, monai_anisotropy]
        # 'other_functions' from original now effectively just 'blur' if 'motion' (grid_distortion) is always applied
        other_monai_functions = [monai_blur] 

        selected_monai_transforms = []
        if aug_level == 1:
            selected_functions_constructors = random.sample(all_monai_functions, 1)
        elif aug_level == 2:
            # Ensure there are enough elements to sample from
            if not blur_monai_functions or not other_monai_functions:
                 selected_functions_constructors = []
            else:
                selected_blur_functions_constructors = random.sample(blur_monai_functions, 1)
                selected_other_functions_constructors = random.sample(other_monai_functions, 1)
                selected_functions_constructors = selected_blur_functions_constructors + selected_other_functions_constructors
        elif aug_level == 3:
            # Handle cases where sampling 2 might exhaust the list
            if len(blur_monai_functions) < 2:
                selected_blur_functions_constructors = blur_monai_functions[:]
            else:
                selected_blur_functions_constructors = random.sample(blur_monai_functions, 2)
            
            if len(other_monai_functions) < 2:
                selected_other_functions_constructors = other_monai_functions[:]
            else:
                selected_other_functions_constructors = random.sample(other_monai_functions, 2)

            selected_functions_set = set(selected_blur_functions_constructors + selected_other_functions_constructors)
            selected_functions_constructors = list(selected_functions_set)
        else:
            selected_functions_constructors = []

        for func_constructor in selected_functions_constructors:
            selected_monai_transforms.append(func_constructor())

        random_compose = Compose(selected_monai_transforms)
        subject = random_compose(subject)

        subject = self.rescale(subject)

        return subject

# --- Placeholder for FDG_Dataset_MONAI (from previous answer) ---
from torch.utils.data import Dataset, DataLoader

class FDG_Dataset_MONAI(Dataset):
    def __init__(self, file_paths, transform=None):
        self.T1_paths = file_paths
        self.transform = transform
        self.loader = LoadImaged(keys="image")

    def __len__(self):
        return len(self.T1_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        T1_path = self.T1_paths[idx]
        data_dict = {"image": T1_path}
        loaded_data = self.loader(data_dict)

        if self.transform:
            augmented_data = self.transform(loaded_data)
        else:
            augmented_data = loaded_data
            if augmented_data["image"].ndim == 2:
                augmented_data["image"] = augmented_data["image"].unsqueeze(0)

        image_tensor = augmented_data['image']
        return image_tensor
# --- End of placeholder ---

