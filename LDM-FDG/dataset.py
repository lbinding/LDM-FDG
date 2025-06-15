#%% Libs 
from pathlib import Path
import torchio as tio
#from monai.data import Dataset
from torch.utils.data import Dataset
import torch 
import nibabel as nib

#%% Dataset
class FDG_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.T1_paths = sorted(Path(data_dir).glob("*.nii.gz"))
        self.transform = transform

    def __len__(self):
        return len(self.T1_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get path to load in 
        T1_path = self.T1_paths[idx]
        
        # Load the T1-weighted MRI image
        T1_img = nib.load(T1_path).get_fdata()

        # Wrap in Subject (Add dimensions for channels and 3D (for augmentations))
        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.as_tensor(T1_img[None, :, :, None])))

        # Apply augmentation
        if self.transform:
            subject = self.transform(subject)
        
        # Extract transformed image tensor
        image_tensor = torch.tensor(subject['image']['data'].squeeze(-1))

        return image_tensor
