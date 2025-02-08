"""
Module for analyzing generalization performance of machine learning models.
"""

from pathlib import Path
import pandas as pd
import numpy as np

def analyze_training_validation_metrics(neps_output_dir):
    """
    Analyzes training and validation metrics across all NePS configurations.
    Both final epoch and mean across all epochs are analyzed.
    
    Args:
        neps_output_dir (str): Path to the NePS output directory containing results
    """
    results_dir = Path(neps_output_dir) / "results"
    print("\nAnalyzing training-validation generalization across all configurations:")
    
    # Create analysis output file
    analysis_file = Path(neps_output_dir).parent / "validation_train_generalization.txt"
    
    all_configs_metrics_final = []  # for final epoch metrics
    all_configs_metrics_mean = []   # for mean across epochs
    
    def log_print(message, file):
        print(message)
        file.write(message + "\n")
    
    with open(analysis_file, "w") as f:
        # Iterate through all configuration directories
        for config_dir in results_dir.glob("config_*"):
            metrics_file = config_dir / "logging" / "metrics.csv"
            if not metrics_file.exists():
                continue
                
            # Read metrics for this configuration
            df = pd.read_csv(metrics_file)
            
            # Get final epoch metrics
            final_epoch = df['epoch'].max()
            final_train = df[(df['epoch'] == final_epoch) & (df['phase'] == 'train')].iloc[0]
            final_val = df[(df['epoch'] == final_epoch) & (df['phase'] == 'val')].iloc[0]
            
            # Calculate mean metrics across all epochs (only for numeric columns)
            numeric_cols = ['loss', 'accuracy', 'precision', 'recall', 'f1']
            mean_train = df[df['phase'] == 'train'][numeric_cols].mean()
            mean_val = df[df['phase'] == 'val'][numeric_cols].mean()
            
            # Calculate metrics for final epoch
            final_metrics = {
                'config_id': config_dir.name,
                'train_acc': final_train['accuracy'],
                'val_acc': final_val['accuracy'],
                'acc_gap': final_train['accuracy'] - final_val['accuracy'],
                'train_loss': final_train['loss'],
                'val_loss': final_val['loss'],
                'loss_gap': final_val['loss'] - final_train['loss'],
                'train_f1': final_train['f1'],
                'val_f1': final_val['f1'],
                'f1_gap': final_train['f1'] - final_val['f1'],
                'train_precision': final_train['precision'],
                'val_precision': final_val['precision'],
                'precision_gap': final_train['precision'] - final_val['precision'],
                'train_recall': final_train['recall'],
                'val_recall': final_val['recall'],
                'recall_gap': final_train['recall'] - final_val['recall']
            }
            
            # Calculate metrics averaged across all epochs
            mean_metrics = {
                'config_id': config_dir.name,
                'train_acc': mean_train['accuracy'],
                'val_acc': mean_val['accuracy'],
                'acc_gap': mean_train['accuracy'] - mean_val['accuracy'],
                'train_loss': mean_train['loss'],
                'val_loss': mean_val['loss'],
                'loss_gap': mean_val['loss'] - mean_train['loss'],
                'train_f1': mean_train['f1'],
                'val_f1': mean_val['f1'],
                'f1_gap': mean_train['f1'] - mean_val['f1'],
                'train_precision': mean_train['precision'],
                'val_precision': mean_val['precision'],
                'precision_gap': mean_train['precision'] - mean_val['precision'],
                'train_recall': mean_train['recall'],
                'val_recall': mean_val['recall'],
                'recall_gap': mean_train['recall'] - mean_val['recall']
            }
            
            all_configs_metrics_final.append(final_metrics)
            all_configs_metrics_mean.append(mean_metrics)
        
        # Convert to DataFrames for analysis
        df_final = pd.DataFrame(all_configs_metrics_final)
        df_mean = pd.DataFrame(all_configs_metrics_mean)
        
        # Print and log final epoch statistics
        log_print(f"Analyzed {len(df_final)} configurations:", f)
        log_print("\n=== Final Epoch Metrics ===", f)
        log_print("\nAverage Metrics Across All Configs (Final Epoch):", f)
        log_print(f"Training Accuracy: {df_final['train_acc'].mean():.2f}% ± {df_final['train_acc'].std():.2f}%", f)
        log_print(f"Validation Accuracy: {df_final['val_acc'].mean():.2f}% ± {df_final['val_acc'].std():.2f}%", f)
        log_print(f"Accuracy Gap (Train-Val): {df_final['acc_gap'].mean():.2f}% ± {df_final['acc_gap'].std():.2f}%", f)
        
        log_print(f"\nTraining Loss: {df_final['train_loss'].mean():.4f} ± {df_final['train_loss'].std():.4f}", f)
        log_print(f"Validation Loss: {df_final['val_loss'].mean():.4f} ± {df_final['val_loss'].std():.4f}", f)
        log_print(f"Loss Gap (Val-Train): {df_final['loss_gap'].mean():.4f} ± {df_final['loss_gap'].std():.4f}", f)
        
        log_print(f"\nTraining F1: {df_final['train_f1'].mean():.4f} ± {df_final['train_f1'].std():.4f}", f)
        log_print(f"Validation F1: {df_final['val_f1'].mean():.4f} ± {df_final['val_f1'].std():.4f}", f)
        log_print(f"F1 Gap (Train-Val): {df_final['f1_gap'].mean():.4f} ± {df_final['f1_gap'].std():.4f}", f)

        log_print(f"\nTraining Precision: {df_final['train_precision'].mean():.4f} ± {df_final['train_precision'].std():.4f}", f)
        log_print(f"Validation Precision: {df_final['val_precision'].mean():.4f} ± {df_final['val_precision'].std():.4f}", f)
        log_print(f"Precision Gap (Train-Val): {df_final['precision_gap'].mean():.4f} ± {df_final['precision_gap'].std():.4f}", f)

        log_print(f"\nTraining Recall: {df_final['train_recall'].mean():.4f} ± {df_final['train_recall'].std():.4f}", f)
        log_print(f"Validation Recall: {df_final['val_recall'].mean():.4f} ± {df_final['val_recall'].std():.4f}", f)
        log_print(f"Recall Gap (Train-Val): {df_final['recall_gap'].mean():.4f} ± {df_final['recall_gap'].std():.4f}", f)
        
        # Print and log mean statistics
        log_print("\n=== Mean Metrics Across All Epochs ===", f)
        log_print("\nAverage Metrics Across All Configs (Mean Across Epochs):", f)
        log_print(f"Training Accuracy: {df_mean['train_acc'].mean():.2f}% ± {df_mean['train_acc'].std():.2f}%", f)
        log_print(f"Validation Accuracy: {df_mean['val_acc'].mean():.2f}% ± {df_mean['val_acc'].std():.2f}%", f)
        log_print(f"Accuracy Gap (Train-Val): {df_mean['acc_gap'].mean():.2f}% ± {df_mean['acc_gap'].std():.2f}%", f)
        
        log_print(f"\nTraining Loss: {df_mean['train_loss'].mean():.4f} ± {df_mean['train_loss'].std():.4f}", f)
        log_print(f"Validation Loss: {df_mean['val_loss'].mean():.4f} ± {df_mean['val_loss'].std():.4f}", f)
        log_print(f"Loss Gap (Val-Train): {df_mean['loss_gap'].mean():.4f} ± {df_mean['loss_gap'].std():.4f}", f)
        
        log_print(f"\nTraining F1: {df_mean['train_f1'].mean():.4f} ± {df_mean['train_f1'].std():.4f}", f)
        log_print(f"Validation F1: {df_mean['val_f1'].mean():.4f} ± {df_mean['val_f1'].std():.4f}", f)
        log_print(f"F1 Gap (Train-Val): {df_mean['f1_gap'].mean():.4f} ± {df_mean['f1_gap'].std():.4f}", f)

        log_print(f"\nTraining Precision: {df_mean['train_precision'].mean():.4f} ± {df_mean['train_precision'].std():.4f}", f)
        log_print(f"Validation Precision: {df_mean['val_precision'].mean():.4f} ± {df_mean['val_precision'].std():.4f}", f)
        log_print(f"Precision Gap (Train-Val): {df_mean['precision_gap'].mean():.4f} ± {df_mean['precision_gap'].std():.4f}", f)

        log_print(f"\nTraining Recall: {df_mean['train_recall'].mean():.4f} ± {df_mean['train_recall'].std():.4f}", f)
        log_print(f"Validation Recall: {df_mean['val_recall'].mean():.4f} ± {df_mean['val_recall'].std():.4f}", f)
        log_print(f"Recall Gap (Train-Val): {df_mean['recall_gap'].mean():.4f} ± {df_mean['recall_gap'].std():.4f}", f)
        
        # Best/Worst generalizing configs based on final epoch
        best_gen_idx = df_final['acc_gap'].abs().idxmin()
        worst_gen_idx = df_final['acc_gap'].abs().idxmax()
        
        log_print("\n=== Best/Worst Generalizing Configurations (Final Epoch) ===", f)
        log_print("\nBest Generalizing Configuration:", f)
        log_print(f"Config ID: {df_final.loc[best_gen_idx, 'config_id']}", f)
        log_print(f"Train Acc: {df_final.loc[best_gen_idx, 'train_acc']:.2f}%", f)
        log_print(f"Val Acc: {df_final.loc[best_gen_idx, 'val_acc']:.2f}%", f)
        log_print(f"Gap: {df_final.loc[best_gen_idx, 'acc_gap']:.2f}%", f)
        
        log_print("\nWorst Generalizing Configuration:", f)
        log_print(f"Config ID: {df_final.loc[worst_gen_idx, 'config_id']}", f)
        log_print(f"Train Acc: {df_final.loc[worst_gen_idx, 'train_acc']:.2f}%", f)
        log_print(f"Val Acc: {df_final.loc[worst_gen_idx, 'val_acc']:.2f}%", f)
        log_print(f"Gap: {df_final.loc[worst_gen_idx, 'acc_gap']:.2f}%", f)
        
        # Detailed metrics for each configuration
        log_print("\n=== Detailed Metrics for All Configurations ===", f)
        for _, row in df_final.sort_values('val_acc', ascending=False).iterrows():
            config_id = row['config_id']
            mean_row = df_mean[df_mean['config_id'] == config_id].iloc[0]
            
            log_print(f"\nConfig: {config_id}", f)
            log_print("Final Epoch Metrics:", f)
            log_print(f"Training Accuracy: {row['train_acc']:.2f}%", f)
            log_print(f"Validation Accuracy: {row['val_acc']:.2f}%", f)
            log_print(f"Accuracy Gap: {row['acc_gap']:.2f}%", f)
            log_print(f"Training Loss: {row['train_loss']:.4f}", f)
            log_print(f"Validation Loss: {row['val_loss']:.4f}", f)
            log_print(f"Loss Gap: {row['loss_gap']:.4f}", f)
            log_print(f"Training F1: {row['train_f1']:.4f}", f)
            log_print(f"Validation F1: {row['val_f1']:.4f}", f)
            log_print(f"F1 Gap: {row['f1_gap']:.4f}", f)
            log_print(f"Training Precision: {row['train_precision']:.4f}", f)
            log_print(f"Validation Precision: {row['val_precision']:.4f}", f)
            log_print(f"Precision Gap: {row['precision_gap']:.4f}", f)
            log_print(f"Training Recall: {row['train_recall']:.4f}", f)
            log_print(f"Validation Recall: {row['val_recall']:.4f}", f)
            log_print(f"Recall Gap: {row['recall_gap']:.4f}", f)
            
            log_print("\nMean Across Epochs:", f)
            log_print(f"Training Accuracy: {mean_row['train_acc']:.2f}%", f)
            log_print(f"Validation Accuracy: {mean_row['val_acc']:.2f}%", f)
            log_print(f"Accuracy Gap: {mean_row['acc_gap']:.2f}%", f)
            log_print(f"Training Loss: {mean_row['train_loss']:.4f}", f)
            log_print(f"Validation Loss: {mean_row['val_loss']:.4f}", f)
            log_print(f"Loss Gap: {mean_row['loss_gap']:.4f}", f)
            log_print(f"Training F1: {mean_row['train_f1']:.4f}", f)
            log_print(f"Validation F1: {mean_row['val_f1']:.4f}", f)
            log_print(f"F1 Gap: {mean_row['f1_gap']:.4f}", f)
            log_print(f"Training Precision: {mean_row['train_precision']:.4f}", f)
            log_print(f"Validation Precision: {mean_row['val_precision']:.4f}", f)
            log_print(f"Precision Gap: {mean_row['precision_gap']:.4f}", f)
            log_print(f"Training Recall: {mean_row['train_recall']:.4f}", f)
            log_print(f"Validation Recall: {mean_row['val_recall']:.4f}", f)
            log_print(f"Recall Gap: {mean_row['recall_gap']:.4f}", f)

    print(f"\nGeneralization analysis saved to: {analysis_file}")

def analyze_validation_test_generalization(neps_output_dir, test_metrics):
    """
    Analyzes generalization between validation and test set for the best NePS configuration.
    
    Args:
        neps_output_dir (str): Path to the NePS output directory containing results
        test_metrics (dict): Dictionary containing test set metrics
    """
    results_dir = Path(neps_output_dir) / "results"
    analysis_file = Path(neps_output_dir).parent / "validation_test_generalization.txt"
    
    def log_print(message, file):
        print(message)
        file.write(message + "\n")
    
    with open(analysis_file, "w") as f:
        # Find the best configuration's metrics file
        best_config_file = Path(neps_output_dir) / "best_loss_with_config_trajectory.txt"
        with open(best_config_file, "r") as bcf:
            lines = bcf.readlines()
            best_config_id = None
            for line in lines:
                if line.startswith("Config ID:"):
                    best_config_id = line.replace("Config ID:", "").strip()
            
            if best_config_id is None:
                raise ValueError("Could not find Config ID in file")
        
        metrics_file = results_dir / f"config_{best_config_id}" / "logging" / "metrics.csv"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found at {metrics_file}")
            
        # Read validation metrics for best configuration
        df = pd.read_csv(metrics_file)
        final_epoch = df['epoch'].max()
        final_val = df[(df['epoch'] == final_epoch) & (df['phase'] == 'val')].iloc[0]
        
        # Calculate generalization gaps
        log_print("\n=== Validation to Test Set Generalization Analysis ===", f)
        log_print(f"\nBest Configuration (ID: {best_config_id})", f)
        
        # Accuracy
        val_acc = final_val['accuracy']
        test_acc = test_metrics['accuracy']
        acc_gap = val_acc - test_acc
        log_print(f"\nAccuracy:", f)
        log_print(f"Validation: {val_acc:.2f}%", f)
        log_print(f"Test: {test_acc:.2f}%", f)
        log_print(f"Gap (Val-Test): {acc_gap:.2f}%", f)
        
        # Loss
        val_loss = final_val['loss']
        test_loss = test_metrics['loss']
        loss_gap = test_loss - val_loss
        log_print(f"\nLoss:", f)
        log_print(f"Validation: {val_loss:.4f}", f)
        log_print(f"Test: {test_loss:.4f}", f)
        log_print(f"Gap (Test-Val): {loss_gap:.4f}", f)
        
        # F1 Score
        val_f1 = final_val['f1']
        test_f1 = np.mean(test_metrics['f1'])
        f1_gap = val_f1 - test_f1
        log_print(f"\nF1 Score:", f)
        log_print(f"Validation: {val_f1:.4f}", f)
        log_print(f"Test: {test_f1:.4f}", f)
        log_print(f"Gap (Val-Test): {f1_gap:.4f}", f)
        
        # Precision
        val_precision = final_val['precision']
        test_precision = np.mean(test_metrics['precision'])
        precision_gap = val_precision - test_precision
        log_print(f"\nPrecision:", f)
        log_print(f"Validation: {val_precision:.4f}", f)
        log_print(f"Test: {test_precision:.4f}", f)
        log_print(f"Gap (Val-Test): {precision_gap:.4f}", f)
        
        # Recall
        val_recall = final_val['recall']
        test_recall = np.mean(test_metrics['recall'])
        recall_gap = val_recall - test_recall
        log_print(f"\nRecall:", f)
        log_print(f"Validation: {val_recall:.4f}", f)
        log_print(f"Test: {test_recall:.4f}", f)
        log_print(f"Gap (Val-Test): {recall_gap:.4f}", f)

    print(f"\nValidation-Test generalization analysis saved to: {analysis_file}")
