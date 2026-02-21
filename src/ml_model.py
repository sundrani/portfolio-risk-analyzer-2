from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


@dataclass(frozen=True)
class MLModelResult:
    model: DecisionTreeClassifier
    report: str


def generate_synthetic_risk_dataset(n: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Features: [annual_vol, max_drawdown_abs, var_abs, beta, sharpe]
    vol = rng.uniform(0.03, 0.45, size=n)
    mdd = rng.uniform(0.02, 0.60, size=n)
    var = rng.uniform(0.002, 0.06, size=n)
    beta = rng.uniform(0.0, 2.5, size=n)
    sharpe = rng.uniform(-1.0, 2.5, size=n)

    X = np.vstack([vol, mdd, var, beta, sharpe]).T

    # Labeling rule (synthetic "ground truth")
    risk_raw = (
        0.35 * (vol / 0.35) +
        0.35 * (mdd / 0.50) +
        0.15 * (var / 0.04) +
        0.10 * (beta / 2.0) +
        0.05 * ((1.5 - sharpe) / 2.0)
    )
    risk_raw = np.clip(risk_raw, 0, 1)

    y = np.where(risk_raw < 0.35, 0, np.where(risk_raw < 0.70, 1, 2))
    return X, y


def train_decision_tree(seed: int = 42) -> MLModelResult:
    X, y = generate_synthetic_risk_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=25,
        random_state=seed
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, target_names=["Conservative", "Moderate", "Aggressive"])
    return MLModelResult(model=clf, report=report)
