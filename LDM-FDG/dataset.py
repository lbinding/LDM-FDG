# In dataset.py

#%% Libs
from pathlib import Path
import torchio as tio
from torch.utils.data import Dataset
import torch
import nibabel as nib

#%% Dataset
class FDG_Dataset(Dataset):
    """
    A PyTorch Dataset to load NIfTI files from a provided list of file paths.
    """
    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths (list of Path objects): List of paths to the NIfTI files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.T1_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.T1_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get path to load in
        T1_path = self.T1_paths[idx]

        # Load the T1-weighted MRI image
        # Using memmap=False can prevent potential file locking issues
        T1_img = nib.load(T1_path, memmap=False).get_fdata()

        # Wrap in Subject (Add dimensions for channels and 3D (for augmentations))
        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.as_tensor(T1_img[None, :, :, None])))

        # Apply augmentation
        if self.transform:
            subject = self.transform(subject)

        # Extract transformed image tensor
        image_tensor = subject['image']['data'].squeeze(-1) # Squeeze the dummy depth dimension

        return image_tensor