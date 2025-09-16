import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from utils.rese_utils import  load_dataset, create_ReSE
from utils.preprocess_utils import extract_person_id

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EPOCHS = int(os.getenv("EPOCHS", 100))
LR = int(os.getenv("LR", 3e-4))
NUM_FOLDS = int(os.getenv("NUM_FOLDS", 3))

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# Adjusted training function for SampleCNN (1D raw audio) with person-level K-Fold
def train_and_evaluate(model_fn, dataset, num_classes, dataset_name, k_folds=NUM_FOLDS):
    sample_to_person_id = {}
    person_id_to_samples = {}
    
    for i, audio_path in enumerate(dataset.audio_paths):
        person_id = extract_person_id(audio_path)
        if person_id:
            sample_to_person_id[i] = person_id
            if person_id not in person_id_to_samples:
                person_id_to_samples[person_id] = []
            person_id_to_samples[person_id].append(i)
        else:
            print(f"Warning: Could not extract person ID from {audio_path}. Skipping this sample for grouping.")

    unique_person_ids = list(person_id_to_samples.keys())
    if not unique_person_ids:
        print("Error: No unique person IDs found. Cannot perform person-level K-Fold.")
        return {"accuracy": [], "precision": [], "recall": [], "f1": []}, None

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    model = None

    for fold, (train_person_indices, val_person_indices) in enumerate(kf.split(unique_person_ids)):
        print(f"\n🔄 Fold {fold + 1}/{k_folds}")

        train_person_ids = [unique_person_ids[i] for i in train_person_indices]
        val_person_ids = [unique_person_ids[i] for i in val_person_indices]

        train_idx = []
        for p_id in train_person_ids:
            train_idx.extend(person_id_to_samples[p_id])
        
        val_idx = []
        for p_id in val_person_ids:
            val_idx.extend(person_id_to_samples[p_id])

        assert not set(train_idx).intersection(set(val_idx)), "Overlap between train and validation samples detected!"

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)

        
        
        train_labels = [dataset.labels[i] for i in train_idx]
        class_weights_np = compute_class_weight('balanced', classes=np.arange(num_classes), y=np.array(train_labels))
        class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        print(f"Class weights for training fold {fold + 1}: {class_weights_tensor.cpu().numpy()}")

        model = model_fn(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        # Initialize lists to collect loss values for plotting
        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for signals, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
                signals, labels = signals.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * signals.size(0)
            
            epoch_loss = running_loss / len(train_loader.sampler)
            train_losses.append(epoch_loss)
            print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

            # Validation loss for current epoch
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.to(device)
                    outputs = model(signals)
                    val_loss = criterion(outputs, labels)
                    val_running_loss += val_loss.item() * signals.size(0)
            val_epoch_loss = val_running_loss / len(val_loader.sampler)
            val_losses.append(val_epoch_loss)
            print(f"Epoch {epoch+1} Validation Loss: {val_epoch_loss:.4f}")

        # After fold training: plot learning curves
        plt.figure(figsize=(10,6))
        plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
        plt.title(f'Learning Curve Fold {fold+1} - {dataset_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_save_path = f"results/plots/no_overlap_ReSE_{dataset_name}_fold{fold+1}_learning_curve.png"
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path)
        plt.close()
        print(f"Learning curve saved to {plot_save_path}")

        # Validation metrics (last epoch)
        model.eval()
        all_preds, all_labels = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for signals, labels in tqdm(val_loader, desc="Validating"):
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item() * signals.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_loader.sampler)

        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
        val_recall = recall_score(all_labels, all_preds, average="weighted") * 100
        val_f1 = f1_score(all_labels, all_preds, average="weighted") * 100

        fold_metrics["accuracy"].append(val_acc)
        fold_metrics["precision"].append(val_precision)
        fold_metrics["recall"].append(val_recall)
        fold_metrics["f1"].append(val_f1)

        print(f"📊 Fold {fold+1} - Validation Loss: {val_epoch_loss:.4f} | Accuracy: {val_acc:.2f}% | Precision: {val_precision:.2f}% | Recall: {val_recall:.2f}% | F1 Score: {val_f1:.2f}%")

        model_save_path = f"/models/no_overlap_ReSE_{fold+1}_{dataset_name}.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    for metric, values in fold_metrics.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"\n📌 Average {metric}: {avg:.2f}% ± {std:.2f}%")

    return fold_metrics, model

# Main function for 1D CNN (unchanged)
def main():
    paths = [
        os.getenv("SEGMENTS_PATH") + "/1AS",
    ]

    for p in paths:
        print(f"\n🚀 Processing dataset: {p}")
        dataset = load_dataset(p)
        dataset_name = p.split("/")[-1]
        print(f"Dataset: {dataset_name}") 
        num_classes = len(dataset.classes)

        fold_metrics, model = train_and_evaluate(create_ReSE, dataset, num_classes, dataset_name)

        model_save_path = f"/Rese_models/no_overlap_ReSE_{dataset_name}.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Final model for {dataset_name} saved to {model_save_path}")

        results = {
            "metric": ["accuracy", "precision", "recall", "f1"],
            "average": [np.mean(fold_metrics["accuracy"]), np.mean(fold_metrics["precision"]), np.mean(fold_metrics["recall"]), np.mean(fold_metrics["f1"])],
            "std_dev": [np.std(fold_metrics["accuracy"]), np.std(fold_metrics["precision"]), np.std(fold_metrics["recall"]), np.std(fold_metrics["f1"])]
        }

        df = pd.DataFrame(results)
        results_save_path = f"Rese_results/metrics_ReSE_{dataset_name}.csv"
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
        df.to_csv(results_save_path, index=False)
        print(f"\nResults saved to {results_save_path}")
        print(df)


if __name__ == "__main__":
    main()