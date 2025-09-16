import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from utils.svm_utils import train_svm, evaluate_svm
from utils.data_utils import load_train_features
from dotenv import load_dotenv
load_dotenv()

def train_and_evaluate(X, y, groups, save_prefix="models/svm"):
    gkf = GroupKFold(n_splits=3)
    accs, f1s, precs, recs = [], [], [], []

    pbar = tqdm(enumerate(gkf.split(X, y, groups), 1), total=3, desc='Folds')
    for fold, (train_idx, val_idx) in pbar:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train SVM
        clf = train_svm(X_train, y_train, save_path=f"{save_prefix}_fold{fold}.pkl")

        # Evaluate
        metrics, _ = evaluate_svm(clf, X_val, y_val)

        accs.append(metrics["accuracy"])
        f1s.append(metrics["f1"])
        precs.append(metrics["precision"])
        recs.append(metrics["recall"])

        pbar.set_postfix({
            'Acc': f"{metrics['accuracy']:.4f}",
            'F1': f"{metrics['f1']:.4f}",
            'Prec': f"{metrics['precision']:.4f}",
            'Rec': f"{metrics['recall']:.4f}"
        })

    return {
        "accuracy": (np.mean(accs), np.std(accs)),
        "f1": (np.mean(f1s), np.std(f1s)),
        "precision": (np.mean(precs), np.std(precs)),
        "recall": (np.mean(recs), np.std(recs))
    }


# ------------------- MAIN -------------------
if __name__ == "__main__":
    path = os.getenv("ACOUSTIC_FEATURES_PATH")  # Path to the CSV file with features
    # Step 1: Load data
    X, y, groups = load_train_features(path)

    # Step 2: Train & evaluate
    results = {path: train_and_evaluate(X, y, groups, save_prefix="models/svm")}

    # Step 3: Summary
    print("\n=== Summary of Metrics ===\n")
    df_summary = pd.DataFrame({
        "Path": [os.path.basename(p) for p in results.keys()],
        "Accuracy": [f"{m['accuracy'][0]*100:.2f} ± {m['accuracy'][1]*100:.2f}" for m in results.values()],
        "F1 Score": [f"{m['f1'][0]*100:.2f} ± {m['f1'][1]*100:.2f}" for m in results.values()],
        "Precision": [f"{m['precision'][0]*100:.2f} ± {m['precision'][1]*100:.2f}" for m in results.values()],
        "Recall": [f"{m['recall'][0]*100:.2f} ± {m['recall'][1]*100:.2f}" for m in results.values()],
    })

    os.makedirs("results", exist_ok=True)
    df_summary.to_csv("results/svm_results.csv", index=False)
    print(df_summary)