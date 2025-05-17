import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd


def roc_auc_multiclass(labels, probabilities):
    """
    Compute the multi-class ROC AUC using the One-Versus-Rest approach.

    Parameters:
    - labels: (955,) array with true class indices (values from 0 to 108)
    - probabilities: (955, 109) array with predicted probabilities for each class

    Returns:
    - Macro-averaged ROC AUC score
    """
    num_classes = probabilities.shape[1]  # Should be 109 classes
    labels_one_hot = np.eye(num_classes)[labels]  # Convert labels to one-hot encoding (955, 109)

    aucs = []  # List to store AUC for each class

    for i in range(num_classes):
        # True labels for class i (binary: 1 if true class, 0 otherwise)
        y_true = labels_one_hot[:, i]  # Shape: (955,)

        # Predicted probabilities for class i
        y_score = probabilities[:, i]  # Shape: (955,)

        # **Check if class i is missing in the batch**
        num_positives = np.sum(y_true)
        num_negatives = len(y_true) - num_positives

        if num_positives == 0 or num_negatives == 0:
            aucs.append(0.5)  # If only one class is present, set AUC to 0.5 (random chance)
            continue  # Skip further computation

        # Sort by predicted score (descending order)
        sorted_indices = np.argsort(-y_score)
        y_true_sorted = y_true[sorted_indices]

        # Compute TPR and FPR
        cum_positive = np.cumsum(y_true_sorted)
        cum_negative = np.cumsum(1 - y_true_sorted)

        TPR = cum_positive / num_positives  # True positive rate
        FPR = cum_negative / num_negatives  # False positive rate

        auc = np.trapz(TPR, FPR)  # Compute AUC using trapezoidal rule
        aucs.append(auc)

    return np.mean(aucs)  # Macro-averaged AUC over all classes

def compute_metrics_top_k(eval_pred, k=3, KNN=False):
    """
    Compute accuracy, top-K accuracy, and ROC AUC for multi-class classification.

    Parameters:
    - eval_pred: Tuple (logits, labels), where:
      - logits: (N, num_classes) array of raw model outputs (before softmax)
      - labels: (N,) array of true class indices
    - k: Number of top predictions to consider for accuracy

    Returns:
    - Dictionary with accuracy, top-K accuracy, and ROC AUC
    """
    logits, labels = eval_pred  # Unpack logits (raw scores) and true labels
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)  # Softmax

    # Top-1 (standard accuracy)
    predictions = np.argmax(logits, axis=-1)  # Get class with highest probability
    acc = accuracy_score(labels, predictions)  # Standard accuracy

    # Top-K Accuracy Calculation
    top_k_predictions = np.argsort(-probabilities, axis=-1)[:, :k]  # Get top-K predicted classes
    top_k_correct = np.any(top_k_predictions == labels[:, None], axis=-1)  # Check if true label is in top-K
    top_k_acc = np.mean(top_k_correct)  # Compute top-K accuracy

     # Top-5 Accuracy (Calculate-Ability of Classes)
    top_5_predictions = np.argsort(-probabilities, axis=-1)[:, :5]
    top_5_correct = np.any(top_5_predictions == labels[:, None], axis=-1)
    top_5_acc = np.mean(top_5_correct)

    # ROC AUC Calculation
    if not KNN:
        auc = roc_auc_multiclass(labels, probabilities)  # Compute multi-class AUC
    else:
        auc = None

    # F1-score Calculation
    f1 = f1_score(labels, predictions, average="weighted")  # Standard F1-score

    return {
        "accuracy": acc,
        f"top_{k}_accuracy": top_k_acc,
        "top_5_accuracy": top_5_acc,
        "roc_auc": auc,
        "f1": f1,
    }


if __name__ == "__main__":
    y_test = []
    df_stat = pd.DataFrame({"Naive Bayes": compute_metrics_top_k(([], y_test)),
           "Decision Tree": compute_metrics_top_k(([].toarray(), y_test)),
           "Support Vector": compute_metrics_top_k(([].toarray(), y_test)),})
           #"KNN": compute_metrics_top_k((predictions_knn.toarray(), y_test), KNN=True)})


    # Define the metrics to plot
    metrics = ["accuracy", "top_3_accuracy", "top_5_accuracy", "f1", "roc_auc"]
    metrics = ["accuracy",  "top_5_accuracy", "roc_auc"]
    df_selected = df_stat.loc[metrics]

    # Plot grouped bar chart
    plt.figure(figsize=(15, 6))
    bar_width = 0.13  # Width of each bar
    x = np.arange(len(metrics))  # X locations for the bars

    colors = ['blue', 'green', 'yellow', 'orange', 'red', 'purple']
    models = df_selected.columns

    for i, model in enumerate(models):
        plt.bar(x + i * bar_width, df_selected[model], width=bar_width, label=model, color=colors[i])

    # Formatting the plot
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Comparison Across Models")
    plt.xticks(x + bar_width, metrics, rotation=30, ha='right')
    plt.yticks(np.arange(0.1, 1.05, 0.05))
    plt.ylim(0.1, 1)
    plt.legend(title="Models")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.show()