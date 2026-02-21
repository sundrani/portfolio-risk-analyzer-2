from __future__ import annotations

import os
import json
import numpy as np

def save_feature_importance(model, feature_names, out_path: str) -> None:
    """
    Saves feature importance to JSON (works for tree-based models like RandomForest).
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature_importances_")

    importances = model.feature_importances_.tolist()
    ranked = sorted(
        [{"feature": f, "importance": float(i)} for f, i in zip(feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ranked, f, indent=2)


def save_shap_summary_plot(model, X, feature_names, out_png_path: str) -> None:
    """
    Saves SHAP summary plot as PNG. Uses TreeExplainer for tree-based models.
    """
    import shap
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    # TreeExplainer works well for RandomForest / tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values can be list (multiclass) -> pick overall display
    plt.figure()
    if isinstance(shap_values, list):
        # for multiclass, show the mean absolute across classes
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        shap.summary_plot(mean_abs, features=X, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, features=X, feature_names=feature_names, show=False)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=180)
    plt.close()