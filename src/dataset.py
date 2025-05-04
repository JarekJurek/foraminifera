import os
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from skimage.io import imread


class ForamsDataset(Dataset):

    def __init__(self, 
                 csv_labels_path=None, 
                 labelled_data_path=None, 
                 unlabeled_data_path=None, 
                 volume_transforms=None,
                 max_num_samples=None):
        
        self.volume_transforms = volume_transforms
        self.max_num_samples = max_num_samples
        self.data = []

        if csv_labels_path and labelled_data_path:
            self.load_labelled_data(csv_labels_path, labelled_data_path)
        if unlabeled_data_path:
            self.load_unlabelled_data(unlabeled_data_path)

        print(f"[INFO] Loaded {len(self.data)} total samples.")

    def _reached_sample_limit(self):
        return self.max_num_samples and len(self.data) >= self.max_num_samples

    def load_labelled_data(self, csv_labels_path, labelled_data_path):
        labels_df = pd.read_csv(csv_labels_path)

        for volume_filename in tqdm(os.listdir(labelled_data_path), desc='Loading labelled data'):
            if self._reached_sample_limit():
                break

            volume_path = os.path.join(labelled_data_path, volume_filename)
            volume = imread(volume_path)

            volume_id = volume_filename.split('_')[2]
            volume_labels = labels_df[labels_df['id'].apply(lambda x: x.split('_')[1]) == volume_id]
            if volume_labels.empty:
                continue

            label = volume_labels['label'].values[0]
            self.data.append({
                'volume': volume,
                'label': label
            })

    def load_unlabelled_data(self, unlabeled_data_path):
        for volume_filename in tqdm(os.listdir(unlabeled_data_path), desc='Loading unlabelled data'):
            if self._reached_sample_limit():
                break

            volume_path = os.path.join(unlabeled_data_path, volume_filename)
            volume = imread(volume_path)

            self.data.append({
                'volume': volume,
                'label': -1
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        volume = self.data[idx]['volume']

        if volume.ndim == 4 and volume.shape[0] == 2:
            print(f"[FIXED] Volume at idx={idx} has 2 channels, taking only the first one")
            volume = volume[0]

        volume = volume.astype(np.float32) / 255.0
        if self.volume_transforms:
            volume = self.volume_transforms(volume)

        volume = torch.tensor(volume).unsqueeze(0)  # Add channel dimension: [1, D, H, W]
        label = torch.tensor(self.data[idx]['label'], dtype=torch.long)

        return volume, label, idx

    def plot_sample_slices(self, idx, idx_x=None, idx_y=None, idx_z=None, cmap='gray', save_path=None):
        volume, label = self[idx]

        if idx_x is None:
            idx_x = volume.shape[3] // 2
        if idx_y is None:
            idx_y = volume.shape[2] // 2
        if idx_z is None:
            idx_z = volume.shape[1] // 2

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        axes[0].imshow(volume[0, :, :, idx_x], cmap=cmap)
        axes[0].set_title(f'Slice at X={idx_x}')
        axes[0].axis('off')

        axes[1].imshow(volume[0, :, idx_y, :], cmap=cmap)
        axes[1].set_title(f'Slice at Y={idx_y}')
        axes[1].axis('off')

        axes[2].imshow(volume[0, idx_z, :, :], cmap=cmap)
        axes[2].set_title(f'Slice at Z={idx_z}')
        axes[2].axis('off')

        plt.suptitle(f'Label: {label}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
    

class Forams2DStackedDataset(Dataset):
    def __init__(self, csv_labels_path, labelled_data_path, slice_transforms=None, max_num_samples=None):
        self.labelled_data_path = labelled_data_path
        self.slice_transforms = slice_transforms
        self.max_num_samples = max_num_samples

        self.volume_paths = []
        self.labels = []

        labels_df = pd.read_csv(csv_labels_path)

        for volume_filename in tqdm(os.listdir(labelled_data_path), desc="Indexing volumes"):
            volume_path = os.path.join(labelled_data_path, volume_filename)

            # Get volume ID and corresponding label
            volume_id = volume_filename.split('_')[2]
            label = labels_df[labels_df['id'].apply(lambda x: x.split('_')[1]) == volume_id]['label'].values[0]

            self.volume_paths.append(volume_path)
            self.labels.append(label)

            if self.max_num_samples and len(self.volume_paths) >= self.max_num_samples:
                break

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        volume = imread(self.volume_paths[idx]).astype(np.float32) / 255.0
        label = self.labels[idx]

        # Get central slice along each axis
        z_idx, y_idx, x_idx = [dim // 2 for dim in volume.shape]
        slice_z = volume[z_idx, :, :]
        slice_y = volume[:, y_idx, :]
        slice_x = volume[:, :, x_idx]

        # Stack into (C=3, H, W)
        stacked_slices = np.stack([slice_z, slice_y, slice_x], axis=0)  # shape [3, H, W]

        if self.slice_transforms:
            stacked_slices = self.slice_transforms(torch.tensor(stacked_slices))

        return stacked_slices.float(), torch.tensor(label, dtype=torch.long)

    

if __name__ == "__main__":
    pass
            
            

            
