"""Model explanation helpers (CatBoost feature contributions)."""

from __future__ import annotations

from typing import Any

import numpy as np

from bin2img.features import FeatureVector
from bin2img.model import MalwareFamilyClassifier


def explain_prediction(
    classifier: MalwareFamilyClassifier,
    features: FeatureVector | np.ndarray,
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    """Return top-k feature contributions for the predicted class."""
    if not classifier.is_fitted or classifier._model is None:
        return {"method": "none", "top_features": []}

    if isinstance(features, FeatureVector):
        row = features.as_row().reshape(1, -1)
        names = list(features.names)
    else:
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        names = list(classifier.feature_names)

    model = classifier._model
    method = "shap"
    try:
        from catboost import Pool

        pool = Pool(row, feature_names=names)
        shap = np.asarray(model.get_feature_importance(type="ShapValues", data=pool))
        if shap.ndim == 3:
            proba = np.asarray(model.predict_proba(row))[0]
            cls_idx = int(np.argmax(proba))
            vals = shap[0, cls_idx, :-1]
        else:
            vals = shap[0, :-1]
        contrib = [(names[i], float(vals[i])) for i in range(min(len(names), len(vals)))]
    except Exception:  # noqa: BLE001
        method = "global_importance_x_value"
        try:
            imp = np.asarray(model.get_feature_importance(), dtype=np.float64)
        except Exception:  # noqa: BLE001
            return {"method": "none", "top_features": []}
        scaled = imp * np.abs(row[0][: len(imp)])
        contrib = [
            (names[i], float(scaled[i])) for i in range(min(len(names), len(scaled)))
        ]

    contrib.sort(key=lambda t: abs(t[1]), reverse=True)
    top = [
        {"feature": name, "contribution": value} for name, value in contrib[:top_k]
    ]
    return {"method": method, "top_features": top}
