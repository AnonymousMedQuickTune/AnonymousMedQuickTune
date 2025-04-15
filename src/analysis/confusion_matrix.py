import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(conf_matrix, metrics, class_names, save_path):
    """
    Creates and saves a confusion matrix visualization with performance metrics.

    Args:
        conf_matrix (np.ndarray): The confusion matrix to plot
        metrics (dict): Dictionary containing various metrics (e.g., accuracy, precision, etc.)
        class_names (list): List of class names for labels
        save_path (Path): Path where to save the confusion matrix plot
    """
    num_classes = len(class_names)

    # Adjust figure size based on number of classes
    fig_size = max(8, num_classes * 1.5)
    plt.figure(figsize=(fig_size, fig_size))

    # Create heatmap with dynamic class labels
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Format metrics text dynamically
    metrics_text = ""

    # Process all metrics except confusion matrix
    for metric_name, metric_value in metrics.items():
        if metric_name != "confusion_matrix":
            if isinstance(metric_value, (list, np.ndarray)):
                # Handle per-class metrics
                metrics_text += f"\n{metric_name.capitalize()}:\n"
                for i, class_name in enumerate(class_names):
                    metrics_text += f"{class_name}: {metric_value[i]*100:.2f}%\n"
            else:
                # Handle global metrics
                metrics_text += f"{metric_name.capitalize()}: {metric_value:.2f}%\n"

    # Adjust text position based on figure size
    plt.figtext(0.02, -0.15 - (num_classes * 0.02), metrics_text, fontsize=10)

    # Save with tight layout to include metrics
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300, format="pdf")
    plt.close()
