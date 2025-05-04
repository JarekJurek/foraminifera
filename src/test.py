import torch
import torchvision.transforms as T
import os
import pandas as pd
from torch.utils.data import DataLoader
from src.models.resnet_3d import Resnet3D
from src.dataset import ForamsDataset


def test(model, unlabelled_dataset, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_ids = []

    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=8, shuffle=False)

    for batch in unlabelled_loader:
        x, _, ids = batch
        x = x.to(device)

        with torch.no_grad():
            outputs = model.model(x)
            probs = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probs, dim=1)

        all_preds.extend(predicted_classes.cpu().tolist())

        for i in range(len(ids)):
            all_ids.append(ids[i] if isinstance(ids[i], str) else ids[i].item())

    df_unlabelled = pd.DataFrame({
        'id': all_ids,
        'label': all_preds
    })

    return df_unlabelled


def main():
    data_path = "/dtu/3d-imaging-center/courses/02510/data/Foraminifera/kaggle_data/"
    model_path = "/zhome/a2/c/213547/group_Anhinga/foraminifera/trained_models/3D_prop_2.pth"

    unlabelled_path = os.path.join(data_path, 'volumes', 'unlabelled')
    unlabelled_csv_path = os.path.join(data_path, 'unlabelled.csv')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Resnet3D(num_classes=15, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.model.load_state_dict(checkpoint)

    unlabelled_dataset = ForamsDataset(
        csv_labels_path=unlabelled_csv_path,
        unlabeled_data_path=unlabelled_path,
        max_num_samples=None
    )

    df_unlabelled = test(model, unlabelled_dataset, device)

    output_csv_path = "3D_prop_submission.csv"
    df_unlabelled.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")


if __name__ == "__main__":
    main()
