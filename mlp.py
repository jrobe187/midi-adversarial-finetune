import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys

def get_args_parser():
    parser = argparse.ArgumentParser("Embedding classifier")

    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--embedding_ext", type=str, default="auto",
                        choices=["auto", "npy", "pt"])
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--nb_classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--label_col", type=str, required=True)
    parser.add_argument("--filename_col", type=str, default="filename")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument('--model_type', type=str, choices=['mlp','linear'])
    parser.add_argument("--hidden1", type=int, default=1024,
                        help="Size of first hidden layer")
    parser.add_argument("--hidden2", type=int, default=512,
                        help="Size of second hidden layer")
    parser.add_argument("--output_path", type=str, default="best_model.pth",
                    help="Path to save best model checkpoint")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load --checkpoint and evaluate on the test split")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved state_dict to load (required if --eval_only)")
    return parser

from sklearn.metrics import roc_auc_score

def evaluate(model, loader, criterion, device, nb_classes):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for emb_batch, label_batch in loader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(emb_batch)
            total_loss += criterion(logits, label_batch).item()
            correct += (logits.argmax(1) == label_batch).sum().item()
            total += len(label_batch)
            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(label_batch.cpu())

    avg_loss = total_loss / len(loader)
    acc = correct / total
    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    auc = float("nan")
    if len(np.unique(y_true)) >= 2:
        try:
            if nb_classes == 2:
                auc = roc_auc_score(y_true, probs[:, 1])
            else:
                auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[warn] AUC failed: {e}")
    return avg_loss, acc, auc

def detect_embedding_ext(embedding_dir):
    for f in os.listdir(embedding_dir):
        if f.endswith(".pt"):
            return "pt"
        elif f.endswith(".npy"):
            return "npy"
    raise FileNotFoundError(f"No .pt or .npy files found in {embedding_dir}")

def load_embedding(path, ext):
    if ext == "pt":
        return torch.load(path)
    elif ext == "npy":
        return torch.tensor(np.load(path), dtype=torch.float32)

def build_mlp(embed_dim, hidden1, hidden2, nb_classes):
    return nn.Sequential(
        nn.Linear(embed_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, nb_classes)
    )

def build_linear_probe(embed_dim, nb_classes):
    return nn.Linear(embed_dim, nb_classes)


def get_criterion(nb_classes):
    return nn.CrossEntropyLoss()

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    #args.output_path = args.output_path + f'/best_model_{args.hidden1}_{args.hidden2}.pth'
    if args.embedding_ext == "auto":
        args.embedding_ext = detect_embedding_ext(args.embedding_dir)
    print(f"Using embedding format: {args.embedding_ext}")

    # Load CSV and match to embeddings
    df = pd.read_csv(args.csv_path)
    embeddings, labels, missing = [], [], 0

    for _, row in df.iterrows():
        base = os.path.splitext(row[args.filename_col])[0]
        emb_path = os.path.join(args.embedding_dir, f"{base}.{args.embedding_ext}")
        if not os.path.exists(emb_path):
            missing += 1
            continue
        embeddings.append(load_embedding(emb_path, args.embedding_ext))
        labels.append(int(row[args.label_col]))

    if missing > 0:
        print(f"Warning: {missing} embeddings not found and skipped")

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"Loaded {len(embeddings)} embeddings, shape {embeddings.shape}")

    # Stratified split
    indices = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size,
        stratify=labels.numpy(), random_state=args.seed
    )

    train_loader = DataLoader(
        EmbeddingDataset(embeddings[train_idx], labels[train_idx]),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        EmbeddingDataset(embeddings[test_idx], labels[test_idx]),
        batch_size=args.batch_size, shuffle=False
    )
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Model, loss, optimizer

    if args.model_type == 'mlp':
        mlp = build_mlp(args.embed_dim, args.hidden1, args.hidden2, args.nb_classes).to(device)
    elif args.model_type == 'linear':
        mlp = build_linear_probe(args.embed_dim, args.nb_classes).to(device)
    else:
        print("Please input correct model type")
        sys.exit(0)

    if args.eval_only:
        if args.checkpoint is None:
            print("--eval_only requires --checkpoint")
            sys.exit(1)
        state = torch.load(args.checkpoint, map_location=device)
        mlp.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")

        criterion = get_criterion(args.nb_classes)
        test_loss, test_acc, test_auc = evaluate(mlp, test_loader, criterion, device, args.nb_classes)
        print("\n--- Evaluation (loaded checkpoint) ---")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")
        sys.exit(0)

    print(f"MLP: {mlp}")
    criterion = get_criterion(args.nb_classes)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience)

    # Best metrics tracking
    best_test_loss = float("inf")
    best_test_acc = 0.0
    best_test_auc = float("nan")
    best_train_loss = 0.0
    best_train_acc = 0.0
    best_train_auc = float("nan")
    best_epoch = 0

    # Training loop
    for epoch in range(args.epochs):
        # ---- Train (gradient step pass)
        mlp.train()
        for emb_batch, label_batch in train_loader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = mlp(emb_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

        # ---- Train metrics (loss/acc/AUC), evaluated post-update like the test pass
        train_loss, train_acc, train_auc = evaluate(mlp, train_loader, criterion, device, args.nb_classes)

        # ---- Test
        test_loss, test_acc, test_auc = evaluate(mlp, test_loader, criterion, device, args.nb_classes)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f} | "
              f"Val Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}")

        # Track best
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_acc = test_acc
            best_test_auc = test_auc
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_train_auc = train_auc
            best_epoch = epoch + 1
            torch.save(mlp.state_dict(), args.output_path)

        # ---- Early stopping
        early_stopping.step(test_loss)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\n--- Best Results (Epoch {best_epoch}) ---")
    print(f"Train Loss: {best_train_loss:.4f} | Train Acc: {best_train_acc:.4f} | Train AUC: {best_train_auc:.4f}")
    print(f"Test Loss:   {best_test_loss:.4f} | Val Acc:   {best_test_acc:.4f} | Val AUC: {best_test_auc:.4f}")
