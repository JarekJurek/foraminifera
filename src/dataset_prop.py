import os
import pandas as pd
from tifffile import imread
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ForamsDatasetProp(Dataset):
    def __init__(self,
                 true_labels_csv=None,
                 prop_labels_csv=None,
                 labelled_data_path=None,
                 unlabelled_data_path=None,
                 volume_transforms=None,
                 max_num_samples=None):
        
        self.volume_transforms = volume_transforms
        self.max_num_samples = max_num_samples
        self.data = []

        self.true_labels_dict = self._load_labels_dict(true_labels_csv) if true_labels_csv else {}
        self.prop_labels_dict = self._load_labels_dict(prop_labels_csv) if prop_labels_csv else {}

        if labelled_data_path:
            self.load_labelled_data(labelled_data_path)

        if unlabelled_data_path:
            self.load_unlabelled_data(unlabelled_data_path)

    def _load_labels_dict(self, csv_path):
        df = pd.read_csv(csv_path)
        df['id'] = df['id'].astype(str)
        return dict(zip(df['id'], df['label']))

    def load_labelled_data(self, labelled_data_path):
        for volume_filename in tqdm(os.listdir(labelled_data_path), desc='Loading labelled data'):
            volume_path = os.path.join(labelled_data_path, volume_filename)
            volume_id = Path(volume_filename).stem  # 'labelled_foram_00001_sc_0'

            if volume_id not in self.true_labels_dict:
                continue

            volume = imread(volume_path)
            label = self.true_labels_dict[volume_id]

            self.data.append({
                'volume': volume,
                'label': label
            })

            if self.max_num_samples and len(self.data) >= self.max_num_samples:
                break

    def load_unlabelled_data(self, unlabelled_data_path):
        for volume_filename in tqdm(os.listdir(unlabelled_data_path), desc='Loading unlabelled data'):
            volume_path = os.path.join(unlabelled_data_path, volume_filename)
            volume_id_raw = Path(volume_filename).stem  # 'foram_00001_sc_0'
            volume_id = volume_id_raw.split('_')[1]  # get the index part → '00001'

            int_id = str(int(volume_id))  # remove leading zeros, so '00001' → '1'

            if int_id not in self.prop_labels_dict:
                continue

            volume = imread(volume_path)
            label = self.prop_labels_dict[int_id]

            self.data.append({
                'volume': volume,
                'label': label
            })

            if self.max_num_samples and len(self.data) >= self.max_num_samples:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        volume = self.data[idx]['volume']
        volume = volume.astype(np.float32) / 255.0
        label = self.data[idx]['label']

        if self.volume_transforms:
            volume = self.volume_transforms(volume)

        label = torch.tensor(label, dtype=torch.long)
        return volume, label, idx

    def plot_sample_slices(self, idx, idx_x=None, idx_y=None, idx_z=None, cmap='gray', save_path=None):
        volume, label, _ = self[idx]

        if idx_x is None:
            idx_x = volume.shape[2] // 2
        if idx_y is None:
            idx_y = volume.shape[1] // 2
        if idx_z is None:
            idx_z = volume.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        
        if idx_x is not None:
            axes[0].imshow(volume[:, :, idx_x], cmap=cmap)
            axes[0].set_title(f'Slice at X={idx_x}')
            axes[0].axis('off')
            
        if idx_y is not None:
            axes[1].imshow(volume[:, idx_y, :], cmap=cmap)
            axes[1].set_title(f'Slice at Y={idx_y}')
            axes[1].axis('off')
            
        if idx_z is not None:
            axes[2].imshow(volume[idx_z, :, :], cmap=cmap)
            axes[2].set_title(f'Slice at Z={idx_z}')
            axes[2].axis('off')
        
        plt.suptitle(f'Label: {label}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
