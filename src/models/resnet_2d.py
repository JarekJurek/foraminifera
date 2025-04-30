import pytorch_lightning as pl
import torch
import torchmetrics
from torchvision.models import resnet18
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score


class Resnet2D(pl.LightningModule):

    def __init__(self, num_classes, pretrained=True, learning_rate=1e-4):
        super().__init__()

        self.num_classes = num_classes
        self.model = resnet18(pretrained=pretrained)
        
        # Adjust for 1-channel input (grayscale slices)
        # self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = torch.nn.Conv2d(3, self.model.conv1.out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace FC head
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.model.fc.in_features, num_classes)
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        # Initialize F1 score metric
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average='macro')

    def log_metrics(self, targets, outputs, type='train', batch_size=None):
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)

        # Compute F1 score using torchmetrics
        f1 = self.f1_score(preds, targets)


        metric_dict = {
            f'{type}_loss': loss,
            f'{type}_accuracy': torchmetrics.functional.accuracy(preds, targets, task='multiclass', average='macro', num_classes=self.num_classes),
            f'{type}_precision': torchmetrics.functional.precision(preds, targets, task='multiclass', average='macro', num_classes=self.num_classes),
            f'{type}_recall': torchmetrics.functional.recall(preds, targets, task='multiclass', average='macro', num_classes=self.num_classes),
            f'{type}_f1': f1,
        }

        self.log_dict(
            metric_dict,
            on_epoch=True, on_step=False,
            prog_bar=True, logger=True,
            batch_size=batch_size
        )
        return metric_dict

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        metrics = self.log_metrics(targets, outputs, type='train', batch_size=len(batch[0]))
        return metrics['train_loss']

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        metrics = self.log_metrics(targets, outputs, type='val', batch_size=len(batch[0]))
        return metrics['val_loss']

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        metrics = self.log_metrics(targets, outputs, type='test', batch_size=len(batch[0]))
        return metrics['test_loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer

    def inference(self, image_tensor):
        """
        Perform inference on a single 2D image tensor.
        Input shape: (1, H, W)
        """
        self.eval()
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))  # add batch dim
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        return predicted_class, probabilities
