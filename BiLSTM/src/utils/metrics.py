from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }