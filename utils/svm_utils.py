import os
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ------------------- Train & Save SVM -------------------
def train_svm(X_train, y_train, save_path="models/svm_model.pkl"):
    """
    Train an SVM classifier and save the model.
    """
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)
    clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"✅ SVM model saved to {save_path}")
    return clf

# ------------------- Evaluate SVM -------------------
def evaluate_svm(clf, X_val, y_val):
    """
    Evaluate an SVM model on validation data.
    """
    y_pred = clf.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred)
    }
    return metrics, y_pred

# ------------------- Load & Predict -------------------
def load_svm_model(path):
    """
    Load a trained SVM model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found at {path}")
    clf = joblib.load(path)
    return clf

def predict_with_svm(clf, X):
    """
    Use a trained SVM model for predictions.
    """
    return clf.predict(X)
