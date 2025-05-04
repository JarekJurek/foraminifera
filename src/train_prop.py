from src.models.MAE import MAE_3D_Lightning
from src.models.resnet_3d import Resnet3D
from src.dataset_prop import ForamsDatasetProp
from src.utils import *

import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight


TRAINED_MODELS_DIR = "trained_models/"
if not os.path.exists(TRAINED_MODELS_DIR):
    os.makedirs(TRAINED_MODELS_DIR)
    print(f"Created directory: {TRAINED_MODELS_DIR}")

DATA_PATH = "/dtu/3d-imaging-center/courses/02510/data/Foraminifera/kaggle_data/"

TRAIN_SPLIT = 0.8
NUM_SAMPLES = None
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8

LEARNING_RATE = 1e-4
BATCH_SIZE = 8


def train(model_pl, train_dataloader, val_dataloader=None, model_name='network'):

    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"accelerator: {accelerator}")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss' if val_dataloader is not None else 'train_loss',
        mode='min',
        patience=EARLY_STOPPING_PATIENCE
    )

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="foramifiera-self-supervised",
        name=f"{model_name}_{wandb.util.generate_id()}",  # Create a unique run name
        log_model=True  # This will log your model checkpoints
    )
    # Optional: Log hyperparameters
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
    
    # Train the model
    trainer.fit(model_pl, train_dataloader, val_dataloader)
    
    save_path = f"{TRAINED_MODELS_DIR}/{model_name}.pth"

    # Save the model
    print(f"Saving the model to {save_path}")
    torch.save(model_pl.model.state_dict(), save_path)

    # Finish the wandb run
    wandb.finish()
    
    return model_pl


if __name__ == "__main__":
    # augmentations for training only
    train_transforms = transforms.Compose([
        RandomFlip3D(),
        RandomRotation3D(),
        RandomGaussianNoise(),
        ToTensorAndChannelFirst()
    ])

    # simple transforms for validation
    val_transforms = transforms.Compose([
        ToTensorAndChannelFirst()
    ])

    dataset = ForamsDatasetProp(
        true_labels_csv=os.path.join(DATA_PATH, "labelled.csv"), 
        prop_labels_csv="/zhome/a2/c/213547/group_Anhinga/foraminifera/new_labells_raw.csv", 
        labelled_data_path=os.path.join(DATA_PATH, "volumes", "labelled"),
        unlabelled_data_path=os.path.join(DATA_PATH, "volumes", "unlabelled"),
        volume_transforms=None,
        max_num_samples=NUM_SAMPLES
    )

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # manually assign transforms - both subsets share the same dataset
    train_dataset.dataset.volume_transforms = train_transforms
    val_dataset.dataset.volume_transforms = val_transforms

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weights from the training labels
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    unique_classes = np.unique(train_labels)
    class_weights_array = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
    class_weights = torch.ones(15, dtype=torch.float)  # Default weights = 1.0
    class_weights[unique_classes] = torch.tensor(class_weights_array, dtype=torch.float)

    model_pl = Resnet3D(num_classes=15, pretrained=True, learning_rate=LEARNING_RATE, class_weights=class_weights)

    train(model_pl, train_loader, val_loader, model_name='3D_prop_2')
