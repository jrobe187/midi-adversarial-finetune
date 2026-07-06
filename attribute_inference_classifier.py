import argparse
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
import torch
import pickle
import pickle as pk
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import quantile_transform


def stratified_split(y, train_frac, val_frac, test_frac, seed):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    idx = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1 - train_frac),
        stratify=y,
        random_state=seed
    )

    y_temp = y[temp_idx]
    val_ratio = val_frac / (val_frac + test_frac)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=seed
    )

    return train_idx, val_idx, test_idx


def _load_embedding_file(fp: Path) -> np.ndarray:
    try:
        obj = torch.load(fp, map_location="cpu", weights_only=False)

        if torch.is_tensor(obj):
            t = obj
        elif isinstance(obj, dict):
            t = None
            for k in ("embedding", "emb", "feat", "features", "x"):
                v = obj.get(k, None)
                if torch.is_tensor(v):
                    t = v
                    break
            if t is None:
                raise ValueError(f"Dict in {fp} has no recognized tensor key: {list(obj.keys())}")
        else:
            t = torch.as_tensor(obj)

        if hasattr(t, "is_quantized") and t.is_quantized:
            t = t.dequantize()

        return t.detach().to(torch.float32).cpu().numpy().ravel()

    except (pickle.UnpicklingError, RuntimeError, EOFError, IsADirectoryError):
        pass
    except ValueError:
        raise

    z = np.load(fp, allow_pickle=False)

    if isinstance(z, np.lib.npyio.NpzFile):
        key = None
        for k in z.files:
            v = z[k]
            if isinstance(v, np.ndarray) and v.size > 0 and v.dtype != object:
                key = k
                break
        if key is None:
            key = z.files[0]
        arr = z[key]
    else:
        arr = z

    return np.asarray(arr, dtype=np.float32).ravel()


def load_dataset(input_csv: str, dataset_path: str, y_label: str):
    df = pd.read_csv(input_csv)

    ids = [Path(x).stem for x in df["filename"].astype(str).to_list()]
    y = df[y_label].to_numpy()

    base = Path(dataset_path)
    X_list = []

    for stem in ids:
        fp = base / f"{stem}.npy"
        if not fp.exists():
            continue
        arr = _load_embedding_file(fp)
        X_list.append(arr)

    X = np.stack(X_list, axis=0)
    return X, y


def main():
    ap = argparse.ArgumentParser(description='run classifier on airogs')
    ap.add_argument('--input_csv')
    ap.add_argument('--y_label')
    ap.add_argument('--dataset_path')
    ap.add_argument('--grid_search', action='store_true')
    ap.add_argument('--normalization', action='store_true')
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    scoring = ''
    if args.y_label == 'gender_numeric':
        scoring = "roc_auc"
    elif args.y_label == 'age_range':
        scoring = "roc_auc_ovr"

    X, y = load_dataset(args.input_csv, args.dataset_path, args.y_label)
    X = np.asarray(X)
    y = np.asarray(y)

    print("Total elements:", X.size)
    print("NaN count:", np.isnan(X).sum())
    print("Inf count:", np.isinf(X).sum())

    bad_rows = np.where(~np.isfinite(X).all(axis=1))[0]
    print("Rows with any NaN/Inf:", len(bad_rows))
    print("First few bad row indices:", bad_rows[:10])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(args.seed))

    train_auc_scores = []
    test_auc_scores = []

    for run in range(args.runs):
        run_seed = args.seed + run

        if args.normalization:
            clf = make_pipeline(
                QuantileTransformer(output_distribution="normal"),
                LogisticRegression(class_weight="balanced", max_iter=500, solver="saga"),
            )
        else:
            clf = make_pipeline(
                LogisticRegression(class_weight="balanced", max_iter=500, solver="saga"),
            )

        train_idx, val_idx, test_idx = stratified_split(y, 0.6, 0.2, 0.2, seed=run_seed)
        tv_idx = np.concatenate([train_idx, val_idx])
        X_tv, y_tv = X[tv_idx], y[tv_idx]

        if not args.grid_search:
            clf.fit(X_tv, y_tv)

            train_prob = clf.predict_proba(X[tv_idx])
            test_prob = clf.predict_proba(X[test_idx])

            if args.y_label == 'gender_numeric':
                train_auc = roc_auc_score(y[tv_idx], train_prob[:, 1])
                test_auc = roc_auc_score(y[test_idx], test_prob[:, 1])
            elif args.y_label == 'age_range':
                train_auc = roc_auc_score(y[tv_idx], train_prob, multi_class="ovr")
                test_auc = roc_auc_score(y[test_idx], test_prob, multi_class="ovr")

        else:
            param_grid = [
                {
                    "logisticregression__C": [0.01, 0.1, 1.0, 10.0],
                    "logisticregression__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
                }
            ]

            grid = GridSearchCV(
                clf,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1
            )

            grid.fit(X_tv, y_tv)

            print(f"Run {run+1} — Best params: {grid.best_params_}, Best CV score: {grid.best_score_:.4f}")

            clf = grid.best_estimator_

            train_prob = clf.predict_proba(X[tv_idx])
            test_prob = clf.predict_proba(X[test_idx])

            if args.y_label == 'gender_numeric':
                train_auc = roc_auc_score(y[tv_idx], train_prob[:, 1])
                test_auc = roc_auc_score(y[test_idx], test_prob[:, 1])
            elif args.y_label == 'age_range':
                train_auc = roc_auc_score(y[tv_idx], train_prob, multi_class="ovr")
                test_auc = roc_auc_score(y[test_idx], test_prob, multi_class="ovr")

        print(f"Run {run+1} — Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
        train_auc_scores.append(train_auc)
        test_auc_scores.append(test_auc)

    mean_train_auc = np.mean(train_auc_scores)
    mean_test_auc = np.mean(test_auc_scores)
    std_test_auc = np.std(test_auc_scores)

    print(f"\nTRAIN AUC AVERAGED OVER {args.runs} RUNS: {mean_train_auc:.4f}")
    print(f"TEST AUC AVERAGED OVER {args.runs} RUNS: {mean_test_auc:.4f} ± {std_test_auc:.4f}")


if __name__ == "__main__":
    main()
