from src.models.resnet_2d import Resnet2D
from src.dataset import Forams2DSliceDataset  

import torchvision.transforms as T
import pytorch_lightning as pl
import torch
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Subset

TRAINED_MODELS_DIR = f"{DATA_PATH}/trained_models/"
if not os.path.exists(TRAINED_MODELS_DIR):
    os.makedirs(TRAINED_MODELS_DIR)
    print(f"Created directory: {TRAINED_MODELS_DIR}")

DATA_PATH = "/zhome/a2/c/213547/group_Anhinga/forams_classification/data"

TRAIN_SPLIT = 0.8
NUM_SAMPLES = 800
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 6

LEARNING_RATE = 1e-4
BATCH_SIZE = 32  # 2D is much lighter, bigger batch size possible!

def train(model_pl, train_dataloader, val_dataloader=None, model_name='network'):

    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"accelerator: {accelerator}")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss' if val_dataloader is not None else 'train_loss',
        mode='min',
        patience=EARLY_STOPPING_PATIENCE
    )

    wandb_logger = WandbLogger(
        project="foramifiera-2d-classification",
        name=f"resnet2d_{wandb.util.generate_id()}",
        log_model=True
    )
    
    wandb_logger.log_hyperparams({
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'num_samples': NUM_SAMPLES,
    })
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        accelerator=accelerator,
        logger=wandb_logger
    )
    
    trainer.fit(model_pl, train_dataloader, val_dataloader)

    save_path = f"{TRAINED_MODELS_DIR}/{model_name}.pth"
    print(f"Saving the model to {save_path}")
    torch.save(model_pl.model.state_dict(), save_path)

    wandb.finish()
    
    return model_pl


if __name__ == "__main__":

    # 2D augmentations for training
    train_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.GaussianBlur(kernel_size=3),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Simple transforms for validation
    val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the 2D slices dataset
    dataset = Forams2DStackedDataset(
    csv_labels_path=os.path.join(DATA_PATH, "labelled.csv"),
    labelled_data_path=os.path.join(DATA_PATH, "volumes/volumes", "labelled"),  # <== now points to raw volumes!
    slice_transforms=None,
    max_num_samples=NUM_SAMPLES
    )

    # Split into train and val
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # Assign transforms manually
    train_dataset.dataset.slice_transforms = train_transforms
    val_dataset.dataset.slice_transforms = val_transforms

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Initialize the 2D model (e.g., ResNet18)
    model_pl = Resnet2D(num_classes=15, pretrained=True, learning_rate=LEARNING_RATE)

    # Train
    train(model_pl, train_loader, val_loader, model_name='resnet2d_normal')
