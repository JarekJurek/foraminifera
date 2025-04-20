from torch.utils.data import DataLoader
from dataset import ForamsDataset
import os
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def extract_features(data_loader, model):
    """
    Extract features from the 3D CNN model.
    :param data_loader: DataLoader object
    :param model: 3D CNN model
    :return: features and labels
    """
    features, labels = [], np.empty(0)
    with torch.no_grad():
        for inputs in tqdm(data_loader, desc='Extracting features'):
            # Each input contains (volume, label)
            volume_inputs_batch = inputs[0].float().to(device)
            labels_batch = inputs[1].to(device).cpu().numpy()
            
            # Forward pass to get features
            output = model(volume_inputs_batch)
            output = output.squeeze().cpu().numpy()
            
            # Append the features to the list
            features.append(output)
            labels = np.append(labels, labels_batch)
            
    return np.vstack(features), labels

# 1. Load the data
DATA_PATH = '../data/'
    
labelled_path = os.path.join(DATA_PATH, 'volumes/volumes/labelled')
unlabelled_path = os.path.join(DATA_PATH, 'volumes/volumes/unlabelled')
labels_csv_path = os.path.join(DATA_PATH, 'labelled.csv')

volume_transforms = transforms.Compose([
    lambda x: torch.from_numpy(x).permute(3, 0, 1, 2) if x.ndim == 4 else torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1, 1),
])

# Load the labelled data
dataset = ForamsDataset(
    csv_labels_path=labels_csv_path, labelled_data_path=labelled_path, 
    volume_transforms=volume_transforms, 
    unlabeled_data_path=unlabelled_path,
    max_num_samples=None
)
x, y = dataset[0]
print(f"Volume shape: {x.shape}")
print(f"Label: {y}")

print(f"Number of samples in dataset: {len(dataset)}")

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2. Extract features using ResNet3D
# Load a pretrained 3D CNN (example with 3D ResNet)
model = r3d_18(pretrained=True)

# Remove the classification layer to get features
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
    
# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
feature_extractor = feature_extractor.to(device)

volume_features, labels = extract_features(dataloader, feature_extractor)
print(f"Extracted features shape: {volume_features.shape}")
print(f"Labels shape: {labels.shape}")

# Save the features and labels
SAVE_FEATURES_PATH = '../features/'
os.makedirs(SAVE_FEATURES_PATH, exist_ok=True)
np.save(os.path.join(SAVE_FEATURES_PATH, 'volume_features.npy'), volume_features)
np.save(os.path.join(SAVE_FEATURES_PATH, 'labels.npy'), labels)

# # 3. Use PCA to reduce dimensionality
# pca = PCA(n_components=2)
# pca.fit(volume_features)
# volume_features_pca = pca.transform(volume_features)
# print(f"Reduced features shape: {volume_features_pca.shape}")
# print(f"Reduced features: {volume_features_pca}")

# SAVE_FIGURE_PATH = '../figures/'

# # Plot the PCA results
# plt.figure(figsize=(8, 6))
# plt.scatter(volume_features_pca[:, 0], volume_features_pca[:, 1], c=labels, cmap='viridis', alpha=0.9)
# plt.colorbar(label='Labels')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('PCA of 3D CNN Features')
# plt.show()
# # Save the figure
# plt.savefig(os.path.join(SAVE_FIGURE_PATH, 'pca_3d_cnn_features.png'))
