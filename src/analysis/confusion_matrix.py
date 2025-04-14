import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(conf_matrix, metrics, save_path):
    """
    Creates and saves a confusion matrix visualization with performance metrics.
    
    Args:
        conf_matrix (np.ndarray): The confusion matrix to plot
        metrics (dict): Dictionary containing accuracy, precision, recall, and f1 metrics
        save_path (Path): Path where to save the confusion matrix plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add metrics as text below the plot
    plt.figtext(0.02, -0.15, 
                f'Accuracy: {metrics["accuracy"]:.2f}%\n'
                f'Precision: {metrics["precision"]:.2f}%\n'
                f'Recall: {metrics["recall"]:.2f}%\n'
                f'F1-Score: {metrics["f1"]:.2f}%',
                fontsize=10)
    
    # Save with tight layout to include metrics
    plt.tight_layout()
    plt.savefig(save_path, 
                bbox_inches='tight', 
                dpi=300, 
                format='pdf')
    plt.close()