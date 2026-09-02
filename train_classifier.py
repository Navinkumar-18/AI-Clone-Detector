"""
train_classifier.py - Downstream Classifier Training & EER Metric Evaluation
=============================================================================

DESIGN RATIONALE & KEY CONCEPTS (FOR REVIEW/DEFENSE):
-----------------------------------------------------
1. WHY TRAIN BOTH LOGISTIC REGRESSION AND A PYTORCH MLP HEAD?
   - Scikit-Learn Logistic Regression: Serves as a fast, linear, deterministic baseline. 
     If a linear classifier performs exceptionally well, it proves the SSL feature space is linearly separable.
   - PyTorch 2-Layer MLP: Adds non-linear expressiveness via ReLU activation and regularizes via Dropout. 
     This allows capturing non-linear interactions between spectral dimensions without overfitting.

2. WHY ACCURACY IS MISLEADING & WHY WE NEED EQUAL ERROR RATE (EER):
   - Imbalanced Datasets: In voice spoofing datasets (like ASVspoof 2019 LA), spoof audio often outnumbers 
     bonafide audio 9:1 (e.g. 22,800 spoof vs 2,580 bonafide in training).
     A dummy classifier predicting "SPOOF" for 100% of samples achieves ~90% Accuracy while being 
     completely useless at protecting bonafide users (100% False Rejection Rate).
   - Security Trade-off: Security systems must balance False Acceptance Rate (FAR - accepting fake voice) 
     and False Rejection Rate (FRR - rejecting real user). 
     Equal Error Rate (EER) finds the decision threshold where FAR == FRR, giving a true, unbiased measure 
     of anti-spoofing security capability regardless of class balance.

3. AUTOMATED SUSPICIOUSLY GOOD EER CHECK:
   - If Eval EER is unusually low (< 2.0%), the script prints an explicit warning to manually inspect 
     splits for speaker ID leakage. In ASVspoof 2019 LA, train/dev/eval speakers must be strictly disjoint.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib

from utils import compute_eer, get_model_tag


# ----------------------------------------------------
# PyTorch MLP Head Architecture
# ----------------------------------------------------
class DeepfakeMLPClassifier(nn.Module):
    """
    Lightweight 2-Layer MLP Classifier Head.
    Architecture: Linear(768 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1)
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, dropout_rate: float = 0.3):
        super(DeepfakeMLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)  # Outputs raw logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate_predictions(y_true: np.ndarray, y_scores: np.ndarray, model_name: str = "Classifier"):
    """
    Compute comprehensive classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, and EER.
    """
    # Decision threshold at 0.5 probability (or 0 logit)
    y_pred = (y_scores >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_scores) * 100.0
    except Exception:
        auc = 50.0
        
    eer, threshold = compute_eer(y_true, y_scores)
    
    print(f"\n---------------- {model_name} Evaluation ----------------")
    print(f" Accuracy:                 {acc:.2f}%")
    print(f" Precision (Bonafide):     {prec*100:.2f}%")
    print(f" Recall (Bonafide):        {rec*100:.2f}%")
    print(f" F1-Score:                 {f1*100:.2f}%")
    print(f" ROC-AUC:                  {auc:.2f}%")
    print(f" Equal Error Rate (EER):   {eer:.2f}% (Threshold: {threshold:.4f})")
    
    # Automated Suspiciously Good EER Check
    if eer < 2.0:
        print("\n [!] WARNING: SUSPICIOUSLY GOOD EER DETECTED (< 2.0%) [!]")
        print("  - Please manually verify that train, dev, and eval splits have NO speaker ID overlap.")
        print("  - Data leakage (e.g. same speaker in train and eval) produces artificially perfect EERs.")
        print("  - Ensure ASVspoof protocol splits are used directly without random shuffling across speakers.")
        
    return {"accuracy": acc, "f1": f1, "auc": auc, "eer": eer, "threshold": threshold}


def train_pytorch_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev: np.ndarray, y_dev: np.ndarray,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3
):
    """
    Train PyTorch MLP with AdamW optimizer, BCEWithLogitsLoss, and early stopping on Dev EER.
    """
    print("\nTraining PyTorch 2-Layer MLP Classifier Head...")
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    dev_dataset = TensorDataset(torch.from_numpy(X_dev).float(), torch.from_numpy(y_dev).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    model = DeepfakeMLPClassifier(input_dim=X_train.shape[1], hidden_dim=128, dropout_rate=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_dev_eer = float('inf')
    best_model_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            
        train_loss = total_loss / len(train_dataset)
        
        # Evaluate on Dev split
        model.eval()
        dev_scores = []
        dev_targets = []
        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits)
                dev_scores.extend(probs.cpu().numpy())
                dev_targets.extend(batch_y.numpy())
                
        dev_scores = np.array(dev_scores)
        dev_targets = np.array(dev_targets)
        dev_eer, _ = compute_eer(dev_targets, dev_scores)
        
        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            best_model_weights = model.state_dict().copy()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f" Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Dev EER: {dev_eer:.2f}% (Best Dev EER: {best_dev_eer:.2f}%)")
            
    # Load best checkpoint
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Classifier Head on Cached SSL Embeddings")
    parser.add_argument("--features_dir", type=str, default="./data/features", help="Directory containing cached .npy features")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory to save trained model checkpoints")
    parser.add_argument("--model_name", type=str, default="microsoft/wavlm-base", help="HuggingFace backbone model ID used for features")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for MLP head")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()
    
    os.makedirs(args.models_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tag = get_model_tag(args.model_name)
    
    print(f"Loading versioned feature embeddings for model tag: '{model_tag}'...")
    
    # Define file paths
    train_x_path = os.path.join(args.features_dir, f"X_train_{model_tag}.npy")
    train_y_path = os.path.join(args.features_dir, f"y_train_{model_tag}.npy")
    dev_x_path   = os.path.join(args.features_dir, f"X_dev_{model_tag}.npy")
    dev_y_path   = os.path.join(args.features_dir, f"y_dev_{model_tag}.npy")
    eval_x_path  = os.path.join(args.features_dir, f"X_eval_{model_tag}.npy")
    eval_y_path  = os.path.join(args.features_dir, f"y_eval_{model_tag}.npy")
    
    # Fallback to demo synthetic feature creation if dataset is not yet downloaded
    if not os.path.exists(train_x_path):
        print(f"\n[!] Notice: Cached feature file '{train_x_path}' not found on disk.")
        print("Creating synthetic feature embeddings to demonstrate classifier pipeline execution...")
        
        np.random.seed(42)
        # Create synthetic 768-dim embeddings
        X_train = np.random.randn(200, 768).astype(np.float32)
        y_train = np.random.choice([0, 1], size=(200,), p=[0.7, 0.3])
        # Add slight mean offset to separate classes synthetically
        X_train[y_train == 1] += 0.4
        
        X_dev = np.random.randn(50, 768).astype(np.float32)
        y_dev = np.random.choice([0, 1], size=(50,), p=[0.7, 0.3])
        X_dev[y_dev == 1] += 0.4
        
        X_eval = np.random.randn(100, 768).astype(np.float32)
        y_eval = np.random.choice([0, 1], size=(100,), p=[0.7, 0.3])
        X_eval[y_eval == 1] += 0.4
    else:
        X_train, y_train = np.load(train_x_path), np.load(train_y_path)
        X_dev, y_dev     = np.load(dev_x_path),   np.load(dev_y_path)
        X_eval, y_eval   = np.load(eval_x_path),  np.load(eval_y_path)
        
    print(f" Loaded Train features: {X_train.shape}, targets: {y_train.shape}")
    print(f" Loaded Dev features:   {X_dev.shape}, targets: {y_dev.shape}")
    print(f" Loaded Eval features:  {X_eval.shape}, targets: {y_eval.shape}")
    
    # ----------------------------------------------------
    # 1. Baseline: Logistic Regression
    # ----------------------------------------------------
    print("\n==================== 1. LOGISTIC REGRESSION BASELINE ====================")
    log_reg = LogisticRegression(max_iter=1000, C=1.0)
    log_reg.fit(X_train, y_train)
    
    log_reg_eval_probs = log_reg.predict_proba(X_eval)[:, 1]
    evaluate_predictions(y_eval, log_reg_eval_probs, model_name="Logistic Regression Baseline")
    
    # Save Logistic Regression checkpoint
    lr_ckpt_path = os.path.join(args.models_dir, f"logistic_regression_{model_tag}.joblib")
    joblib.dump(log_reg, lr_ckpt_path)
    print(f" Saved Logistic Regression checkpoint: {lr_ckpt_path}")
    
    # ----------------------------------------------------
    # 2. PyTorch MLP Classifier Head
    # ----------------------------------------------------
    print("\n==================== 2. PYTORCH 2-LAYER MLP HEAD ====================")
    mlp_model = train_pytorch_mlp(
        X_train, y_train, X_dev, y_dev, 
        device=device, epochs=args.epochs, batch_size=args.batch_size
    )
    
    # Evaluate MLP on Eval Split
    mlp_model.eval()
    with torch.no_grad():
        mlp_logits = mlp_model(torch.from_numpy(X_eval).float().to(device))
        mlp_eval_probs = torch.sigmoid(mlp_logits).cpu().numpy()
        
    mlp_metrics = evaluate_predictions(y_eval, mlp_eval_probs, model_name="PyTorch MLP Classifier Head")
    
    # Save PyTorch MLP Checkpoint
    mlp_ckpt_path = os.path.join(args.models_dir, f"best_mlp_{model_tag}.pt")
    torch.save({
        "model_state_dict": mlp_model.state_dict(),
        "input_dim": X_train.shape[1],
        "hidden_dim": 128,
        "model_name": args.model_name,
        "eer": mlp_metrics["eer"],
        "threshold": mlp_metrics["threshold"]
    }, mlp_ckpt_path)
    print(f" Saved Best PyTorch MLP Checkpoint: {mlp_ckpt_path}")


if __name__ == "__main__":
    main()
