import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from utils.vgg_utils import load_dataset, VGGCBAMClassifier
from utils.preprocess_utils import extract_person_id

# ==============================
# Config
# ==============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
NUM_FOLDS = 3


# ==============================
# Training & Evaluation
# ==============================
def train_and_evaluate(dataset, num_classes, dataset_name, k_folds=NUM_FOLDS):
    sample_to_person_id = {}
    person_id_to_samples = {}
    for i, (path, _) in enumerate(dataset.samples):
        person_id = extract_person_id(path)
        if person_id:
            sample_to_person_id[i] = person_id
            person_id_to_samples.setdefault(person_id, []).append(i)

    unique_person_ids = list(person_id_to_samples.keys())
    print(f"🔍 Found {len(unique_person_ids)} unique person IDs in the dataset.")
    if not unique_person_ids:
        return {"accuracy": [], "precision": [], "recall": [], "f1": []}, None

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    model = None

    for fold, (train_ids, val_ids) in enumerate(kf.split(unique_person_ids)):
        print(f"\n🔄 Fold {fold+1}/{k_folds}")
        train_idx = [i for pid in [unique_person_ids[j] for j in train_ids] for i in person_id_to_samples[pid]]
        val_idx = [i for pid in [unique_person_ids[j] for j in val_ids] for i in person_id_to_samples[pid]]
        assert not set(train_idx).intersection(set(val_idx)), "Overlap detected!"

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx), num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx), num_workers=4, pin_memory=True)

        train_labels = [dataset.samples[i][1] for i in train_idx]
        class_weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(num_classes), y=np.array(train_labels)),
                                     dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = VGGCBAMClassifier(num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        # Training
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader.sampler):.4f}")

        # Validation
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        val_acc = accuracy_score(labels_all, preds) * 100
        val_prec = precision_score(labels_all, preds, average="weighted", zero_division=0) * 100
        val_rec = recall_score(labels_all, preds, average="weighted") * 100
        val_f1 = f1_score(labels_all, preds, average="weighted") * 100

        for metric, value in zip(["accuracy", "precision", "recall", "f1"], [val_acc, val_prec, val_rec, val_f1]):
            fold_metrics[metric].append(value)
        print(f"📊 Fold {fold+1} - Acc: {val_acc:.2f}% | Prec: {val_prec:.2f}% | Rec: {val_rec:.2f}% | F1: {val_f1:.2f}%")

        save_path = f"/home/hamid/pfe/models/no_overlap_VGG16_cbam_fold{fold+1}_{dataset_name}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    for metric, values in fold_metrics.items():
        print(f"📌 Average {metric}: {np.mean(values):.2f}% ± {np.std(values):.2f}%")

    return fold_metrics, model


# ==============================
# Main
# ==============================
def main():
    paths = [
        os.getenv("SPECTROGRAM_PATH") + "/1AS",
        os.getenv("SPECTROGRAM_PATH") + "/1FS",
        os.getenv("SPECTROGRAM_PATH") + "/5AS",
        os.getenv("SPECTROGRAM_PATH") + "/5FS",
    ]
    for p in paths:
        print(f"\n🚀 Processing dataset: {p}")
        dataset = load_dataset(p)
        dataset_name = os.path.basename(p)
        num_classes = len(dataset.classes)

        fold_metrics, _ = train_and_evaluate(dataset, num_classes, dataset_name)

        df = pd.DataFrame({
            "metric": ["accuracy", "precision", "recall", "f1"],
            "average": [np.mean(fold_metrics[m]) for m in ["accuracy", "precision", "recall", "f1"]],
            "std_dev": [np.std(fold_metrics[m]) for m in ["accuracy", "precision", "recall", "f1"]]
        })
        results_save_path = f"results_vgg_cbam/metrics_vgg16{dataset_name}.csv"
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
        df.to_csv(results_save_path, index=False)
        print(f"\nResults saved to {results_save_path}")
        print(df)


if __name__ == "__main__":
    main()
