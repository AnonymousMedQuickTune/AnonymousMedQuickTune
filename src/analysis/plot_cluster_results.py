#!/usr/bin/env python3
"""
Script to plot Cross-Validation results from MedQuickTune log files.
Plots loss and metrics for each outer CV fold across all inner folds.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def find_training_log(experiment_path):
    """
    Find the training log file in the experiment directory.
    
    Args:
        experiment_path (str): Path to experiment directory
        
    Returns:
        str: Path to training log file or None if not found
    """
    experiment_dir = Path(experiment_path)
    
    # Look for cluster_oe directory
    cluster_oe_dir = experiment_dir / 'cluster_oe'
    if cluster_oe_dir.exists():
        # Look for medquicktune log files
        log_files = list(cluster_oe_dir.glob('medquicktune.*.err_out'))
        if log_files:
            return str(log_files[0])  # Return first found log file
    
    # Alternative: look for any .err_out files in experiment directory
    log_files = list(experiment_dir.rglob('*.err_out'))
    if log_files:
        return str(log_files[0])
    
    return None

def find_evaluation_file(experiment_path):
    """
    Find the evaluation results file in the experiment directory.
    
    Args:
        experiment_path (str): Path to experiment directory
        
    Returns:
        str: Path to evaluation file or None if not found
    """
    experiment_dir = Path(experiment_path)
    
    # Look for evaluation_results directory
    eval_dir = experiment_dir / 'evaluation_results'
    if eval_dir.exists():
        # Look for evaluation_output.txt
        eval_file = eval_dir / 'evaluation_output.txt'
        if eval_file.exists():
            return str(eval_file)
    
    # Alternative: look for any evaluation files
    eval_files = list(experiment_dir.rglob('*evaluation*'))
    if eval_files:
        return str(eval_files[0])
    
    return None

def parse_log_file(log_file_path):
    """
    Parse the log file to extract training metrics for each CV fold.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary containing parsed data for each CV fold
    """
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Split content by CV fold markers
    cv_fold_pattern = r"Preloading data for CV fold (\d+)/5"
    cv_fold_matches = list(re.finditer(cv_fold_pattern, content))
    
    cv_data = {}
    
    for i, match in enumerate(cv_fold_matches):
        fold_num = int(match.group(1))
        start_pos = match.start()
        
        # Determine end position (next fold or end of file)
        if i + 1 < len(cv_fold_matches):
            end_pos = cv_fold_matches[i + 1].start()
        else:
            end_pos = len(content)
        
        fold_content = content[start_pos:end_pos]
        
        # Extract training fold information
        training_fold_pattern = r"Training Fold (\d+)/5"
        training_fold_matches = list(re.finditer(training_fold_pattern, fold_content))
        
        fold_metrics = {
            'outer_fold': fold_num,
            'inner_folds': {}
        }
        
        for j, tf_match in enumerate(training_fold_matches):
            inner_fold_num = int(tf_match.group(1)) - 1  # Convert 1-5 to 0-4
            tf_start = tf_match.start()
            
            # Determine end of this training fold
            if j + 1 < len(training_fold_matches):
                tf_end = training_fold_matches[j + 1].start()
            else:
                # For the last training fold, look for the end of this outer CV fold
                # Look for next CV fold marker or end of content
                next_cv_fold = re.search(r"Preloading data for CV fold", fold_content[tf_start:])
                if next_cv_fold:
                    tf_end = tf_start + next_cv_fold.start()
                else:
                    tf_end = len(fold_content)
            
            inner_fold_content = fold_content[tf_start:tf_end]
            
            # Extract epoch data
            epoch_pattern = r"\[.*?\]\[Epoch (\d+)\] (Train|Val)\s*-\s*Loss: ([\d.]+), Acc: ([\d.]+)%, Prec: ([\d.]+)%, Rec: ([\d.]+)%, F1: ([\d.]+)%, AUC: ([\d.]+)%"
            epoch_matches = re.findall(epoch_pattern, inner_fold_content)
            
            # Check if epochs were found
            if len(epoch_matches) == 0:
                print(f"    Warning: No epochs found for Inner Fold {inner_fold_num}")
                print(f"    Content length: {len(inner_fold_content)}")
                print(f"    Content preview: {inner_fold_content[:200]}...")
            
            inner_fold_data = {
                'epochs': [],
                'train_loss': [],
                'train_acc': [],
                'train_prec': [],
                'train_rec': [],
                'train_f1': [],
                'train_auc': [],
                'val_loss': [],
                'val_acc': [],
                'val_prec': [],
                'val_rec': [],
                'val_f1': [],
                'val_auc': []
            }
            
            for epoch_match in epoch_matches:
                epoch_num = int(epoch_match[0])
                fold_type = epoch_match[1]
                loss = float(epoch_match[2])
                acc = float(epoch_match[3])
                prec = float(epoch_match[4])
                rec = float(epoch_match[5])
                f1 = float(epoch_match[6])
                auc = float(epoch_match[7])
                
                if fold_type == 'Train':
                    inner_fold_data['epochs'].append(epoch_num)
                    inner_fold_data['train_loss'].append(loss)
                    inner_fold_data['train_acc'].append(acc)
                    inner_fold_data['train_prec'].append(prec)
                    inner_fold_data['train_rec'].append(rec)
                    inner_fold_data['train_f1'].append(f1)
                    inner_fold_data['train_auc'].append(auc)
                elif fold_type == 'Val':
                    inner_fold_data['val_loss'].append(loss)
                    inner_fold_data['val_acc'].append(acc)
                    inner_fold_data['val_prec'].append(prec)
                    inner_fold_data['val_rec'].append(rec)
                    inner_fold_data['val_f1'].append(f1)
                    inner_fold_data['val_auc'].append(auc)
            
            fold_metrics['inner_folds'][inner_fold_num] = inner_fold_data
        
        # Report epoch counts for each fold
        for fold_idx, fold_data in fold_metrics['inner_folds'].items():
            if len(fold_data['epochs']) > 0:
                print(f"    Inner Fold {fold_idx}: {len(fold_data['epochs'])} epochs")
        
        print(f"  Outer Fold {fold_num}: {len(fold_metrics['inner_folds'])} inner folds")
        
        cv_data[fold_num] = fold_metrics
    
    return cv_data

def parse_evaluation_file(evaluation_file_path):
    """
    Parse the evaluation results file to extract performance metrics.
    
    Args:
        evaluation_file_path (str): Path to the evaluation file
        
    Returns:
        dict: Dictionary containing parsed evaluation data
    """
    with open(evaluation_file_path, 'r') as f:
        content = f.read()
    
    evaluation_data = {
        'outer_folds': {},
        'overall_summary': {}
    }
    
    # Split content by outer CV fold sections
    outer_fold_pattern = r"----- Outer CV Fold (\d+)/5 -----"
    outer_fold_matches = list(re.finditer(outer_fold_pattern, content))
    
    for i, match in enumerate(outer_fold_matches):
        fold_num = int(match.group(1))
        start_pos = match.start()
        
        # Determine end position (next fold or end of file)
        if i + 1 < len(outer_fold_matches):
            end_pos = outer_fold_matches[i + 1].start()
        else:
            end_pos = len(content)
        
        fold_content = content[start_pos:end_pos]
        
        # Extract inner fold results
        inner_fold_pattern = r"=== Evaluating Fold (\d+)/5 ===\nLoss: ([\d.]+)\nAccuracy: ([\d.]+)%\nPrecision: ([\d.]+)%\nRecall: ([\d.]+)%\nF1: ([\d.]+)%\nAuc: ([\d.]+)%"
        inner_fold_matches = re.findall(inner_fold_pattern, fold_content)
        
        fold_data = {
            'inner_folds': {},
            'average_results': {}
        }
        
        # Parse inner fold results
        for inner_match in inner_fold_matches:
            inner_fold_num = int(inner_match[0])
            loss = float(inner_match[1])
            accuracy = float(inner_match[2])
            precision = float(inner_match[3])
            recall = float(inner_match[4])
            f1 = float(inner_match[5])
            auc = float(inner_match[6])
            
            fold_data['inner_folds'][inner_fold_num] = {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc * 100  # Convert to percentage  # TODO @Diane: Update after fixing the evaluation script!
            }
        
        # Extract average results for this outer fold
        avg_pattern = r"=== Average Results Across All Folds ===\nLoss: ([\d.]+)\nAccuracy: ([\d.]+)%\nPrecision: ([\d.]+)%\nRecall: ([\d.]+)%\nF1: ([\d.]+)%\nAuc: ([\d.]+)%"
        avg_match = re.search(avg_pattern, fold_content)
        if avg_match:
            fold_data['average_results'] = {
                'loss': float(avg_match.group(1)),
                'accuracy': float(avg_match.group(2)),
                'precision': float(avg_match.group(3)),
                'recall': float(avg_match.group(4)),
                'f1': float(avg_match.group(5)),
                'auc': float(avg_match.group(6)) * 100  # Convert to percentage # TODO @Diane: Update after fixing the evaluation script!
            }
        
        evaluation_data['outer_folds'][fold_num] = fold_data
    
    # Extract overall summary - use a more flexible approach
    # First try the exact pattern
    overall_pattern = r"=== Average Results Across All Outer CV Folds \(mean\) ===\n\nLoss: ([\d.]+)\nAccuracy: ([\d.]+)%\nPrecision: ([\d.]+)%\nRecall: ([\d.]+)%\nF1: ([\d.]+)%\nAuc: ([\d.]+)%"
    overall_match = re.search(overall_pattern, content)
    
    if not overall_match:
        # Try a more flexible pattern that allows for variations in whitespace
        overall_pattern_flexible = r"=== Average Results Across All Outer CV Folds \(mean\) ===\s*\n\s*Loss:\s*([\d.]+)\s*\n\s*Accuracy:\s*([\d.]+)%\s*\n\s*Precision:\s*([\d.]+)%\s*\n\s*Recall:\s*([\d.]+)%\s*\n\s*F1:\s*([\d.]+)%\s*\n\s*Auc:\s*([\d.]+)%"
        overall_match = re.search(overall_pattern_flexible, content)
    
    if overall_match:
        evaluation_data['overall_summary'] = {
            'loss': float(overall_match.group(1)),
            'accuracy': float(overall_match.group(2)),
            'precision': float(overall_match.group(3)),
            'recall': float(overall_match.group(4)),
            'f1': float(overall_match.group(5)),
            'auc': float(overall_match.group(6)) * 100  # Convert to percentage  # TODO @Diane: Update after fixing the evaluation script!
        }
        print("Successfully extracted overall summary data")
    else:
        print("Warning: Could not extract overall summary data")
    
    return evaluation_data

def create_experiment_folder_structure(experiment_path, base_output_dir):
    """
    Create folder structure based on experiment path.
    
    Args:
        experiment_path (str): Path to experiment directory
        base_output_dir (str): Base output directory
        
    Returns:
        str: Path to the created experiment folder
    """
    experiment_dir = Path(experiment_path)
    
    # Extract experiment information from path
    # Expected structure: experiments/NePS/lipo/test_baseline_densenetv1/seed_42
    path_parts = experiment_dir.parts
    
    try:
        # Find the experiment name and seed
        experiment_name = None
        seed_name = None
        
        for i, part in enumerate(path_parts):
            if part.startswith('test_') or part.startswith('train_'):
                experiment_name = part
                # Look for seed in the next few parts
                for j in range(i+1, min(i+3, len(path_parts))):
                    if path_parts[j].startswith('seed_'):
                        seed_name = path_parts[j]
                        break
                break
        
        if experiment_name and seed_name:
            # Create folder structure: base_output_dir/experiment_name/seed_name/
            output_dir = Path(base_output_dir) / experiment_name / seed_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Created experiment folder structure: {output_dir}")
            return str(output_dir)
        else:
            print("Could not automatically determine experiment structure, using base output directory")
            return base_output_dir
            
    except Exception as e:
        print(f"Error creating folder structure: {e}")
        print("Using base output directory")
        return base_output_dir

def create_individual_folds_plot(fold_data, outer_fold, colors, output_dir):
    """
    Create plot showing individual inner folds for a specific outer fold.
    
    Args:
        fold_data (dict): Data for the specific outer fold
        outer_fold (int): Outer fold number
        colors (list): Color palette for inner folds
        output_dir (str): Output directory
    """
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Outer CV Fold {outer_fold} - Individual Inner Folds', fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    metrics = [
        ('Loss', 'train_loss', 'val_loss'),
        ('Accuracy (%)', 'train_acc', 'val_acc'),
        ('Precision (%)', 'train_prec', 'val_prec'),
        ('Recall (%)', 'train_rec', 'val_rec'),
        ('F1-Score (%)', 'train_f1', 'val_f1'),
        ('AUC (%)', 'train_auc', 'val_auc')
    ]
    
    for idx, (metric_name, train_key, val_key) in enumerate(metrics):
        ax = axes[idx]
        
        for inner_fold_num, inner_fold_data in fold_data['inner_folds'].items():
            if len(inner_fold_data['epochs']) > 0:
                epochs = inner_fold_data['epochs']
                train_vals = inner_fold_data[train_key]
                val_vals = inner_fold_data[val_key]
                
                # Plot training metrics
                ax.plot(epochs, train_vals, 
                       color=colors[inner_fold_num % len(colors)], 
                       linestyle='-', 
                       marker='o', 
                       markersize=4,
                       alpha=0.8,
                       label=f'Inner Fold {inner_fold_num} (Train)')
                
                # Plot validation metrics
                ax.plot(epochs, val_vals, 
                       color=colors[inner_fold_num % len(colors)], 
                       linestyle='--', 
                       marker='s', 
                       markersize=4,
                       alpha=0.8,
                       label=f'Inner Fold {inner_fold_num} (Val)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'outer_fold_{outer_fold}_individual_folds.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved individual folds plot to: {plot_path}")
    
    plt.close()

def create_mean_std_plot(fold_data, outer_fold, output_dir):
    """
    Create plot showing mean and standard deviation across inner folds for a specific outer fold.
    
    Args:
        fold_data (dict): Data for the specific outer fold
        outer_fold (int): Outer fold number
        output_dir (str): Output directory
    """
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Outer CV Fold {outer_fold} - Mean ± Std Across Inner Folds', fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    metrics = [
        ('Loss', 'train_loss', 'val_loss'),
        ('Accuracy (%)', 'train_acc', 'val_acc'),
        ('Precision (%)', 'train_prec', 'val_prec'),
        ('Recall (%)', 'train_rec', 'val_rec'),
        ('F1-Score (%)', 'train_f1', 'val_f1'),
        ('AUC (%)', 'train_auc', 'val_auc')
    ]
    
    for idx, (metric_name, train_key, val_key) in enumerate(metrics):
        ax = axes[idx]
        
        # Collect all epoch data across inner folds
        all_epochs = set()
        for inner_fold_data in fold_data['inner_folds'].values():
            all_epochs.update(inner_fold_data['epochs'])
        
        all_epochs = sorted(list(all_epochs))
        
        # Calculate mean and std for each epoch
        train_means = []
        train_stds = []
        val_means = []
        val_stds = []
        
        for epoch in all_epochs:
            train_vals = []
            val_vals = []
            
            for inner_fold_data in fold_data['inner_folds'].values():
                if epoch in inner_fold_data['epochs']:
                    epoch_idx = inner_fold_data['epochs'].index(epoch)
                    train_vals.append(inner_fold_data[train_key][epoch_idx])
                    val_vals.append(inner_fold_data[val_key][epoch_idx])
            
            if train_vals:
                train_means.append(np.mean(train_vals))
                train_stds.append(np.std(train_vals))
            else:
                train_means.append(np.nan)
                train_stds.append(np.nan)
            
            if val_vals:
                val_means.append(np.mean(val_vals))
                val_stds.append(np.std(val_vals))
            else:
                val_means.append(np.nan)
                val_stds.append(np.nan)
        
        # Plot mean curves with shaded std regions
        # Training metrics
        ax.plot(all_epochs, train_means, 'b-', linewidth=2, label='Train (Mean)', marker='o', markersize=6)
        ax.fill_between(all_epochs, 
                       [m - s for m, s in zip(train_means, train_stds)], 
                       [m + s for m, s in zip(train_means, train_stds)], 
                       alpha=0.3, color='blue', label='Train (±Std)')
        
        # Validation metrics
        ax.plot(all_epochs, val_means, 'r-', linewidth=2, label='Val (Mean)', marker='s', markersize=6)
        ax.fill_between(all_epochs, 
                       [m - s for m, s in zip(val_means, val_stds)], 
                       [m + s for m, s in zip(val_means, val_stds)], 
                       alpha=0.3, color='red', label='Val (±Std)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'outer_fold_{outer_fold}_mean_std.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved mean/std plot to: {plot_path}")
    
    plt.close()

def create_summary_plot(cv_data, output_dir):
    """
    Create a summary plot showing the best validation metrics for each outer fold.
    
    Args:
        cv_data (dict): Parsed CV data
        output_dir (str): Directory to save plots
    """
    print("Creating summary plot...")
    
    # Extract best validation metrics for each outer fold
    summary_data = []
    
    for outer_fold, fold_data in cv_data.items():
        for inner_fold_num, inner_fold_data in fold_data['inner_folds'].items():
            if len(inner_fold_data['val_loss']) > 0:
                # Find best epoch (minimum validation loss)
                best_epoch_idx = np.argmin(inner_fold_data['val_loss'])
                
                summary_data.append({
                    'outer_fold': outer_fold,
                    'inner_fold': inner_fold_num,
                    'best_epoch': inner_fold_data['epochs'][best_epoch_idx],
                    'best_val_loss': inner_fold_data['val_loss'][best_epoch_idx],
                    'best_val_acc': inner_fold_data['val_acc'][best_epoch_idx],
                    'best_val_prec': inner_fold_data['val_prec'][best_epoch_idx],
                    'best_val_rec': inner_fold_data['val_rec'][best_epoch_idx],
                    'best_val_f1': inner_fold_data['val_f1'][best_epoch_idx],
                    'best_val_auc': inner_fold_data['val_auc'][best_epoch_idx]
                })
    
    if not summary_data:
        print("No summary data found!")
        return
    
    df = pd.DataFrame(summary_data)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Best Validation Metrics by Outer CV Fold and Inner Fold', fontsize=16)
    
    axes = axes.flatten()
    
    metrics = [
        ('Loss (lower is better)', 'best_val_loss', True),
        ('Accuracy (%)', 'best_val_acc', False),
        ('Precision (%)', 'best_val_prec', False),
        ('Recall (%)', 'best_val_rec', False),
        ('F1-Score (%)', 'best_val_f1', False),
        ('AUC (%)', 'best_val_auc', False)
    ]
    
    for idx, (metric_name, metric_key, lower_better) in enumerate(metrics):
        ax = axes[idx]
        
        # Group by outer fold and plot inner fold results
        for outer_fold in sorted(df['outer_fold'].unique()):
            fold_data = df[df['outer_fold'] == outer_fold]
            
            x_pos = np.arange(len(fold_data))
            values = fold_data[metric_key].values
            
            # Color code based on performance
            if lower_better:
                colors = ['green' if v == min(values) else 'lightblue' for v in values]
            else:
                colors = ['green' if v == max(values) else 'lightblue' for v in values]
            
            bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Inner Fold')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        inner_folds = sorted(df['inner_fold'].unique())
        ax.set_xticks(range(len(inner_folds)))
        ax.set_xticklabels([f'Fold {f}' for f in inner_folds])
        
        # Add legend for outer folds
        handles = []
        for outer_fold in sorted(df['outer_fold'].unique()):
            handles.append(plt.Rectangle((0,0),1,1, color='lightblue', alpha=0.7, 
                                       label=f'Outer Fold {outer_fold}'))
        ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = Path(output_dir) / 'summary_best_metrics.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to: {summary_path}")
    
    plt.close()
    
    # Save summary data to CSV
    csv_path = Path(output_dir) / 'summary_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary data to: {csv_path}")

def create_plots(cv_data, output_dir="plots"):
    """
    Create plots for each outer CV fold showing metrics across inner folds.
    
    Args:
        cv_data (dict): Parsed CV data
        output_dir (str): Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Colors for different inner folds
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    for outer_fold, fold_data in cv_data.items():
        print(f"Creating plots for Outer CV Fold {outer_fold}")
        
        # Create individual folds plot
        create_individual_folds_plot(fold_data, outer_fold, colors, output_dir)
        
        # Create mean/std plot
        create_mean_std_plot(fold_data, outer_fold, output_dir)
    
    # Create summary plot showing best validation metrics for each outer fold
    create_summary_plot(cv_data, output_dir)

def create_overall_performance_plot(evaluation_data, output_dir):
    """
    Create plot showing overall performance across all outer CV folds.
    
    Args:
        evaluation_data (dict): Parsed evaluation data
        output_dir (str): Output directory
    """    
    overall = evaluation_data['overall_summary']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    values = [overall['accuracy'], overall['precision'], overall['recall'], overall['f1'], overall['auc']]
    
    bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{value:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Performance (%)')
    ax.set_title('Overall Performance Across All Outer CV Folds')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Add loss information
    ax.text(0.02, 0.98, f'Average Loss: {overall["loss"]:.2f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'overall_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved overall performance plot to: {plot_path}")
    
    plt.close()

def create_per_fold_performance_plot(evaluation_data, output_dir):
    """
    Create plot showing performance comparison across outer CV folds.
    
    Args:
        evaluation_data (dict): Parsed evaluation data
        output_dir (str): Output directory
    """
    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Comparison Across Outer CV Folds', fontsize=16)
    
    axes = axes.flatten()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'loss']
    metric_names = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'AUC (%)', 'Loss']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        fold_numbers = []
        fold_values = []
        
        for fold_num, fold_data in evaluation_data['outer_folds'].items():
            if 'average_results' in fold_data and fold_data['average_results']:
                fold_numbers.append(fold_num)
                fold_values.append(fold_data['average_results'][metric])
        
        if fold_numbers:
            bars = ax.bar(fold_numbers, fold_values, color='skyblue', alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, fold_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if metric != 'loss' else -0.1),
                       f'{value:.1f}', ha='center', va='bottom' if metric != 'loss' else 'top', 
                       fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Outer CV Fold')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} by Outer CV Fold')
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits
            if metric != 'loss':
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(0, max(fold_values) * 1.1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'per_fold_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-fold performance plot to: {plot_path}")
    
    plt.close()

def create_detailed_metrics_plot(evaluation_data, output_dir):
    """
    Create detailed metrics comparison plot.
    
    Args:
        evaluation_data (dict): Parsed evaluation data
        output_dir (str): Output directory
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    fold_numbers = sorted(evaluation_data['outer_folds'].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    x = np.arange(len(fold_numbers))
    width = 0.15
    
    # Plot each metric
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = []
        for fold_num in fold_numbers:
            fold_data = evaluation_data['outer_folds'][fold_num]
            if 'average_results' in fold_data and fold_data['average_results']:
                values.append(fold_data['average_results'][metric])
            else:
                values.append(0)
        
        ax.bar(x + i * width, values, width, label=metric_name, alpha=0.8)
    
    ax.set_xlabel('Outer CV Fold')
    ax.set_ylabel('Performance (%)')
    ax.set_title('Detailed Metrics Comparison Across Outer CV Folds')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(fold_numbers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'detailed_metrics_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed metrics comparison plot to: {plot_path}")
    
    plt.close()

def create_evaluation_plots(evaluation_data, output_dir):
    """
    Create plots for evaluation results.
    
    Args:
        evaluation_data (dict): Parsed evaluation data
        output_dir (str): Output directory
    """
    print("Creating evaluation plots...")

    # Create overall performance plot
    create_overall_performance_plot(evaluation_data, output_dir)

    # Create per-fold performance comparison plot
    create_per_fold_performance_plot(evaluation_data, output_dir)
    
    # Create detailed metrics comparison plot
    create_detailed_metrics_plot(evaluation_data, output_dir)
    
    print(f"Successfully created evaluation plots")

def main():
    """
    Main function to process MedQuickTune experiment directories and create visualization plots.
    
    This function:
    1. Parses command line arguments for experiment path and output directory
    2. Automatically finds training log and evaluation files
    3. Creates training plots showing CV fold performance over epochs
    4. Creates evaluation plots showing final performance metrics
    5. Organizes output into training/ and evaluation/ subdirectories
    
    Args:
        experiment_path (str): Path to experiment directory (e.g., experiments/NePS/lipo/test_baseline_densenetv2/seed_43)
        --output-dir (str): Base output directory (default: results)
        --auto-structure (flag): Automatically create folder structure based on experiment path
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If experiment directory doesn't exist
        Exception: For any other processing errors
    """
    parser = argparse.ArgumentParser(description='Plot CV results from MedQuickTune experiment directories')
    parser.add_argument('experiment_path', help='Path to experiment directory (e.g., experiments/NePS/lipo/test_baseline_densenetv2/seed_43)')
    parser.add_argument('--output-dir', default='results', help='Base output directory (default: results)')
    parser.add_argument('--auto-structure', action='store_true', 
                       help='Automatically create folder structure based on experiment path')
    
    args = parser.parse_args()
    
    # Check if experiment directory exists
    if not Path(args.experiment_path).exists():
        print(f"Error: Experiment directory '{args.experiment_path}' not found!")
        return
    
    print(f"Processing experiment: {args.experiment_path}")
    
    try:
        # Find training log file
        training_log = find_training_log(args.experiment_path)
        if not training_log:
            print("Warning: No training log file found!")
            cv_data = {}
        else:
            print(f"Found training log: {training_log}")
            # Parse the training log file
            cv_data = parse_log_file(training_log)
        
        # Find evaluation results file
        evaluation_file = find_evaluation_file(args.experiment_path)
        if not evaluation_file:
            print("Warning: No evaluation results file found!")
            evaluation_data = {}
        else:
            print(f"Found evaluation file: {evaluation_file}")
            # Parse the evaluation file
            evaluation_data = parse_evaluation_file(evaluation_file)
        
        if not cv_data and not evaluation_data:
            print("Error: No data found in either training log or evaluation file!")
            return
        
        # Determine output directory structure
        if args.auto_structure:
            output_dir = create_experiment_folder_structure(args.experiment_path, args.output_dir)
        else:
            output_dir = args.output_dir
        
        # Create training plots if data exists
        if cv_data:
            print(f"Found data for {len(cv_data)} outer CV folds:")
            for fold_num, fold_data in cv_data.items():
                print(f"  Outer Fold {fold_num}: {len(fold_data['inner_folds'])} inner folds")
            
            # Create training subdirectory
            training_dir = Path(output_dir) / 'training'
            training_dir.mkdir(exist_ok=True)
            print(f"Created training directory: {training_dir}")
            
            create_plots(cv_data, str(training_dir))
        
        # Create evaluation plots if data exists
        if evaluation_data:
            # Create evaluation subdirectory
            evaluation_dir = Path(output_dir) / 'evaluation'
            evaluation_dir.mkdir(exist_ok=True)
            print(f"Created evaluation directory: {evaluation_dir}")
            
            create_evaluation_plots(evaluation_data, str(evaluation_dir))
        
        print(f"\nAll plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error processing experiment: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 