"""
facility_classifier.py
======================
A supervised classifier that predicts whether a facility-year will require
compliance review, complementing the unsupervised clustering and anomaly work.

The target, needs_review, is set to 1 when a facility-year either falls below
the EU 70% recovery target OR is flagged as anomalous by the Isolation Forest.
In other words, the actual decision the ComplianceMonitor makes.
Predicting this combined outcome is a genuine supervised task rather than simply
re-learning the anomaly detector, because it fuses the rule-based signal and the
ML signal into the single decision a human reviewer cares about.

Because each facility appears up to five times (once per year), the model is
evaluated with a GROUPED split (GroupKFold by facility): all of a facility's
rows stay together in either training or test, never both. This prevents the
model from simply memorising individual sites and gives an honest estimate of
how it generalises to facilities it has never seen.

Input : out/facility_clusters.csv
Output: out/classifier_results.csv,
        out/figures/classifier_confusion.png,
        out/figures/classifier_importance.png

Run Command:  python facility_classifier.py
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("out")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EU_TARGET = 0.70
FEATURES = ["recovery_share", "disposal_share", "other_share", "backfill_share",
            "hazardous_share", "n_low_codes", "n_records", "log_tonnage"]


def load_and_label():
    """Load facility-years and build the needs_review target label."""
    df = pd.read_csv(OUT / "facility_clusters.csv")
    df["log_tonnage"] = np.log1p(df["total_accepted_t"])

    # Rule signal: recovery below the EU 70% target. recovery_rate is NaN when a
    # facility has no recovery+disposal tonnage (pure storage sites); we treat
    # those as "not below target" so the label captures only genuine shortfalls.
    below_target = (df["recovery_rate"] < EU_TARGET).fillna(False)

    # ML signal: Isolation Forest anomaly flag
    anomaly = df["is_anomaly"].astype(bool)

    df["needs_review"] = (below_target | anomaly).astype(int)

    # facility id (strip the year) for grouped splitting
    df["facility"] = df["licensee"]
    return df


def evaluate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 accuracy_score, confusion_matrix)

    df = load_and_label()
    X = df[FEATURES].values
    y = df["needs_review"].values
    groups = df["facility"].values

    print(f"Dataset: {len(df)} facility-years, {len(df['facility'].unique())} facilities")
    print(f"Target distribution: needs_review={y.sum()} ({y.mean()*100:.1f}%), "
          f"ok={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"Split: GroupKFold by facility (no facility in both train and test)\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6,
                                                class_weight="balanced", random_state=42),
    }

    gkf = GroupKFold(n_splits=5)
    rows = []
    best_rf_importances = None
    agg_conf = {name: np.zeros((2, 2), dtype=int) for name in models}

    for name, model in models.items():
        accs, precs, recs, f1s = [], [], [], []
        for tr, te in gkf.split(X, y, groups):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            model.fit(Xtr, y[tr])
            pred = model.predict(Xte)
            accs.append(accuracy_score(y[te], pred))
            precs.append(precision_score(y[te], pred, zero_division=0))
            recs.append(recall_score(y[te], pred, zero_division=0))
            f1s.append(f1_score(y[te], pred, zero_division=0))
            agg_conf[name] += confusion_matrix(y[te], pred, labels=[0, 1])
            if name == "Random Forest":
                best_rf_importances = model.feature_importances_
        rows.append({
            "model": name,
            "accuracy": np.mean(accs), "accuracy_std": np.std(accs),
            "precision": np.mean(precs), "precision_std": np.std(precs),
            "recall": np.mean(recs), "recall_std": np.std(recs),
            "f1": np.mean(f1s), "f1_std": np.std(f1s),
        })
        print(f"{name}:")
        print(f"    Accuracy  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"    Precision {np.mean(precs):.3f} ± {np.std(precs):.3f}")
        print(f"    Recall    {np.mean(recs):.3f} ± {np.std(recs):.3f}")
        print(f"    F1        {np.mean(f1s):.3f} ± {np.std(f1s):.3f}\n")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "classifier_results.csv", index=False)

    # ---- confusion matrix plot (Random Forest) ----
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})
    cm = agg_conf["Random Forest"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["ok", "needs_review"]); ax.set_yticklabels(["ok", "needs_review"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Random Forest confusion matrix\n(aggregated over 5 folds)")
    plt.tight_layout()
    fig.savefig(FIG / "classifier_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- feature importance plot ----
    if best_rf_importances is not None:
        order = np.argsort(best_rf_importances)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(np.array(FEATURES)[order], best_rf_importances[order], color="#4477aa")
        ax.set_xlabel("Random Forest feature importance")
        ax.set_title("Which features drive the needs_review prediction")
        plt.tight_layout()
        fig.savefig(FIG / "classifier_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {OUT/'classifier_results.csv'}")
    print(f"Saved: {FIG/'classifier_confusion.png'}")
    print(f"Saved: {FIG/'classifier_importance.png'}")
    return results


if __name__ == "__main__":
    evaluate()