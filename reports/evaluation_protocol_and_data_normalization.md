# Evaluation Protocol and Data Normalization

## Nested Cross-Validation



Our evaluation protocol employs a hierarchical validation strategy consisting of three levels:

### 1. Outer Cross-Validation Folds
We use N-repeated stratified K-fold cross-validation (3 repetitions × 5 splits = 15 outer folds) to ensure robust performance estimation through multiple independent train/test splits. The data splitting process uses scikit-learn's `RepeatedStratifiedKFold` with a base seed to generate reproducible splits (automatically shuffles data). Each fold partitions the data into approximately 80% for training/validation and 20% for testing, preserving the original class distribution in both train+val and test sets through stratified sampling. The cross validation splits are cached to disk to ensure consistency across experiments and enable efficient reuse.

**Note:** Currently checking comparison experiments with 1-repeated stratified 5-fold cross validation. 1-repeated stratified 5-fold cross validation should align with Natalia's implementation where the gaps between the outer folds is not that huge.

### 2. Hyperparameter Optimization
Within each outer fold, we perform independent hyperparameter optimization using the NePS (Neural Pipeline Search) and MedQuickTune frameworks. Each outer fold uses a fold-specific random seed (base_seed + fold_index) to ensure different hyperparameter configurations are sampled, preventing identical configurations across outer folds.

### 3. Inner Cross-Validation Folds
For each hyperparameter configuration, we employ stratified K-fold cross-validation (5 folds) within the training data to create ensemble predictions through probability averaging across folds. Each fold partitions the training data into approximately 80% for training and 20% for validation, preserving the original class distribution in both train and validation sets through stratified sampling. The inner fold splitting uses scikit-learn's `StratifiedKFold` with shuffling enabled and a base seed, ensuring that each inner fold maintains the class distribution while providing different train/validation splits for ensemble learning. Training uses early stopping: if the best model from the current inner fold is not improved upon for 20 consecutive epochs, training is terminated to prevent overfitting and reduce computational costs.

### Structure
Our implementation maintains the following hierarchical directory structure:


```
experiment_directory/
├── seed_X/
│   └── NePS_output/
│       ├── cv_outer_fold_0/
│       │   └── configs/
│       │       ├── config_1/
│       │       │   ├── test_evaluation_results.json
│       │       │   └── cv_inner_fold_0/
│       │       │       ├── logging/
│       │       │       │   ├── gradients.csv
│       │       │       │   ├── learning_rates.csv
│       │       │       │   ├── metrics.csv
│       │       │       │   ├── model_info.yaml
│       │       │       │   ├── resources.csv
│       │       │       │   └──timing.csv
│       │       │       ├── best_model_checkpoint.pth
│       │       │       └── normalization_stats.txt         # Only for CT images
│       │       │   └── cv_inner_fold_1/
│       │       │       └── ...
│       │       └── config_2/
│       │           └── ...
│       ├── cv_outer_fold_1/
│       └── ...
```

## Metrics

We calculate comprehensive performance metrics using both macro and micro averaging strategies. Macro averaging assigns equal weight to all classes, making it suitable for imbalanced datasets, while micro averaging weights metrics by class frequency, reflecting overall dataset performance. Our evaluation suite includes accuracy, area under the curve (AUC), precision, recall, F1-score, confusion matrices, and per-class metrics for detailed performance analysis.

## Inner Fold Evaluation

For each hyperparameter configuration, we implement a comprehensive evaluation strategy that provides both ensemble and individual fold analysis:

### Ensemble Evaluation
As our main evaluation protocol is cross-validation ensemble learning where each inner fold trains a separate model and all models predict on the complete test set. Rather than using hard predictions, we average softmax probabilities across folds to create ensemble predictions. This probability-based ensemble approach provides more nuanced predictions and better uncertainty quantification compared to majority voting schemes.

### Per-Fold Evaluation
Additionally, we log comprehensive metrics for each individual inner fold to provide detailed performance analysis and identify fold-specific patterns.

## Outer Fold Evaluation
> **Note:** This is not relevant for portfolio creation

To estimate generalization performance, we use a nested cross-validation protocol with aggregation across all outer folds. In addition, we perform a post-hoc outer-fold ensemble analysis to explore the potential benefits of model ensembling after evaluation.

### Baseline VS HPO runs
For baseline runs, there is just one single configuration for which we perform the evaluation. For HPO runs (NePS/MedQuickTune), we perform the evaluation on the best configuration for each outer fold based on validation performance on the selected main metric.

### Individual Fold Aggregation
As our main evaluation protocol we aggregate results across all outer folds using robust statistical measures. For each metric, we collect values from all outer folds and calculate:
- **Mean ± Standard Deviation** for central tendency and variability measurement
- **Median ± Median Absolute Deviation (MAD)** for robust central tendency and spread measurement
  
This procedure yields an unbiased and statistically valid estimate of generalization performance under nested cross-validation.

### Outer-Fold Ensemble (Post-hoc Aggregation)
In addition to standard aggregation, we perform a post-hoc outer-fold ensemble analysis. Here, softmax probabilities from all models trained in different outer folds are averaged to simulate the effect of combining multiple independently optimized models. This result does not constitute a nested cross-validation because this ensemble is constructed from test set predictions across folds. Possible advantages: variance reduction, improved mean performance, and enhanced robustness.  


## Data Normalization
Data normalization follows the nnU-Net approach to ensure consistent intensity ranges across different imaging modalities. The distinction between CT and MRI datasets is crucial because these modalities have fundamentally different intensity characteristics and acquisition protocols that require tailored normalization strategies:

### MRI datasets:
For MRI datasets, normalization is performed during preprocessing and includes per-image z-score normalization. MRI images exhibit high inter-patient intensity variability due to different scanner settings, coil configurations, and acquisition parameters, making per-image normalization essential for consistent intensity ranges across patients.

### CT datasets:
Each inner fold calculates its own normalization statistics (mean and standard deviation) from its training data only. The normalization includes intensity clipping to [0.5, 99.5] percentiles to remove extreme outliers, followed by z-score normalization. These inner fold specific statistics are then applied to training, validation, and test sets, preventing data leakage. CT images have more consistent intensity ranges across patients (measured in Hounsfield units), allowing for fold-specific normalization that leverages the statistical properties of the training cohort.


**Note 1:** This differs from Natalia's preprocessing script, where all datasets (both CT and MRI) are normalized using per-image z-score normalization during preprocessing.

**Note 2:** If time and compute allow, we could additionally explore **AutoNorm** for 3D Medical Image Classification. In another project, optimizing normalization statistics through hyperparameter optimization instead of calculating them from training data was very successful, though it was only a side experiment that would have needed more extensive testing with additional datasets and models. In this project, AutoNorm could potentially improve performance by learning optimal normalization parameters for each fold through the hyperparameter optimization process. While this would constitute a separate research project, the current codebase is well-suited for exploring this approach due to its modular normalization architecture and integrated hyperparameter optimization framework. 




