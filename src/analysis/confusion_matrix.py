import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(conf_matrix, metrics, class_names, save_path):
    """
    Creates and saves a confusion matrix visualization with performance metrics.
    
    Args:
        conf_matrix (np.ndarray): The confusion matrix to plot
        metrics (dict): Dictionary containing accuracy, precision, recall, and f1 metrics
        class_names (list): List of class names for labels
        save_path (Path): Path where to save the confusion matrix plot
    """
    num_classes = len(class_names)
    
    # Adjust figure size based on number of classes
    fig_size = max(8, num_classes * 1.5)
    plt.figure(figsize=(fig_size, fig_size))
    
    # Create heatmap with dynamic class labels
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Format metrics text
    metrics_text = f'Accuracy: {metrics["accuracy"]:.2f}%\n'
    
    # Add per-class metrics if available
    if isinstance(metrics["precision"], (list, np.ndarray)):
        for i, class_name in enumerate(class_names):
            metrics_text += f'\n{class_name}:\n'
            metrics_text += f'Precision: {metrics["precision"][i]*100:.2f}%\n'
            metrics_text += f'Recall: {metrics["recall"][i]*100:.2f}%\n'
            metrics_text += f'F1-Score: {metrics["f1"][i]*100:.2f}%\n'
    else:
        # Add average metrics if per-class metrics are not available
        metrics_text += f'Precision: {metrics["precision"]:.2f}%\n'
        metrics_text += f'Recall: {metrics["recall"]:.2f}%\n'
        metrics_text += f'F1-Score: {metrics["f1"]:.2f}%'
    
    # Adjust text position based on figure size
    plt.figtext(0.02, -0.15 - (num_classes * 0.02), 
                metrics_text,
                fontsize=10)
    
    # Save with tight layout to include metrics
    plt.tight_layout()
    plt.savefig(save_path, 
                bbox_inches='tight', 
                dpi=300, 
                format='pdf')
    plt.close()