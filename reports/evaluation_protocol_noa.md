# Evaluation Protocol – DEHB Framework

## Data Preprocessing

Before any model optimization or evaluation takes place, all datasets undergo a standardized preprocessing and augmentation procedure to ensure consistency (across training).  
These operations are implemented through a series of MONAI `Compose` transformations defined in the preprocessing module.

The preprocessing is integrated into the workflow at three key moments:

1. **Before cross-validation / DEHB optimization** — data is preprocessed and prepared for each fold.  
2. **During training and validation** — dynamic augmentations are applied to training samples only.  
3. **During testing** — no transformation is applied.

In our implementation, data preprocessing is integrated directly within each training and validation loop rather than performed as a separate preprocessing step.  
For each outer fold, the training and validation data are dynamically loaded and transformed at the beginning of the inner DEHB optimization loop, since preprocessing depends on the specific configuration being evaluated.

In particular, the **resampling strategy**, which is an optimized hyperparameter within the DEHB process, can take one of four discrete options (*mean*, *median*, *isotropic*, or *anisotropic volumetric*).  
To reduce computational costs, a caching mechanism is employed: each resampling option is computed once and subsequently retrieved whenever a configuration requires it, effectively minimizing redundant data transformations during the search.

The preprocessing pipeline itself includes resampling, intensity normalization (modality-specific), region cropping/padding, and data augmentation (random flips, rotations, zooms, and contrast/intensity perturbations).  
Validation and test data do not undergo any transformation.

## Data Partitioning

For every dataset, the available data are first divided into:

- **Training data**, used for optimization and validation. (80%)
- **Held-out test data**, used exclusively for the final evaluation of the best configuration. (20%)

This initial separation ensures that the test set remains completely unseen during model and hyperparameter search.  
Within the training data, we then apply a five-fold stratified cross-validation, forming the outer loop of the evaluation.

## Outer Loop: 5-Fold Cross-Validation

The outer loop provides an estimate of generalization performance through a five-fold cross-validation process:

- In each iteration (fold), 80% of the training data (four folds) are used for model optimization, while the remaining 20% (one fold) serves as a temporary validation fold for that iteration.  
- The `DEHB` object is initiated and goes into an inner loop for optimization and sanity checks if the maximum number of configurations has not yet been reached (10 (*approved*) configurations per fold).

## Inner Loop: Optimization via DEHB

The steps within this inner process are as follows:

1. **Configuration sampling and evaluation:**  
   DEHB proposes configurations along with fidelity levels (training epochs). Each configuration is first *approved* (Sanity check to ensure it can be applied to the dataset) and then used for training and validation on a split of the outer training data through the `objective_function` routine.

2. **Model training and validation:**  
   Each candidate model is trained on the training subset and evaluated on a validation subset to compute ROC-AUC and loss. Early stopping monitors validation loss to prevent unnecessary computation. (Patience: 20)

3. **Result collection:**  
   For every evaluation, DEHB records the validation AUC, loss, fidelity (training budget), and wall-clock runtime.  
   After the search concludes, the best-performing configuration for that fold (lowest cross-entropy loss) is selected.

## Cross-Fold Re-evaluation

After completing the inner optimization for one fold, the best configuration from that fold is **re-evaluated across all other folds**:

- Each optimized configuration is retrained and validated on the remaining folds using their respective train–validation splits.  
- This measures how well a configuration found in one partition performs when applied to others.  
- The results are stored for later use in ensemble methods.

## Aggregation and Best Configuration Selection

After all outer folds are completed, the system averages their results:

- The mean validation AUC and loss across the best-performing configurations are computed.  
- The overall best configuration (i.e., the one with the lowest cross-entropy loss) is selected and saved for final evaluation.  
- The corresponding hyperparameter settings, architecture, and validation statistics are stored for reproducibility.

We have a total of 25 models at the end of the whole training and validation. The main 5: 1 model per fold with the best-performing configuration found (Out of the 10 configurations tested on said fold) and for the 25 models: Each best configuration found on the folds is re-evaluated on the other folds; per configuration, we have a total of 5 models (5 folds).
- 5 Model Ensemble: Ensembles the main 5 models and does an average prediction on the external test-set.
- 25 Model Ensemble: Ensembles the total 25 models and does an average prediction on the external test-set.

## Final Testing Stage (Independent Test Evaluation)

The final evaluation occurs after cross-validation and aggregation are complete.  
This stage uses the held-out test set that was never involved in cross-validation or optimization:

1. The best configuration identified from the training process is retrieved.  
2. The model is retrained from scratch on the entire training data (i.e., all five folds combined).  
3. The retrained model is evaluated once on the independent test set.  
4. The final performance metrics are recorded and reported.

This is the final DEHB AUC score with no ensemble technique. 

# Evaluation Protocol – Baseline (MONAI DenseNet-121)

## Data Preprocessing

For the baseline, the preprocessing pipeline follows the same general structure as in the AutoML framework but with fixed parameters. 
All scans are resampled to a median voxel spacing computed across the dataset, ensuring consistent spatial resolution without optimization.  
After resampling, we perform cropping and padding around the tumor followed by intensity normalization.

During training, standard augmentations (random flips, rotations, and intensity perturbations) are applied to improve robustness, while validation data undergo only normalization.  
Unlike the AutoML framework, no hyperparameters related to preprocessing are tuned, and no caching or configuration-dependent resampling is required.

## Data Partitioning

For each dataset, the available data are divided into five stratified folds.  
At each iteration:

- Four folds (80% of the data) are used for training.  
- The remaining fold (20%) is used for validation.

There is no additional held-out test set used for the baseline; its generalization performance is measured directly from cross-validation, following the same training–validation ratio as the AutoML method’s inner loop.

## Model Configuration

The baseline uses a fixed model and training setup across all folds:

- **Architecture:** `DenseNet121` (3D, 1 input channel, 2 output classes)  
- **Optimizer:** Adam (learning rate = 1×10⁻⁴)  
- **Loss function:** Cross-entropy  
- **Training epochs:** 100 per fold  
- **Batch size:** 1  
- **Data augmentation:** Standard MONAI transformations (random flips, rotations)  
- **Exponential Moving Average (EMA):** β = 0.9999

These hyperparameters remain constant throughout the entire evaluation and are not tuned per dataset or fold.

## Cross-Validation and Metrics

For each of the five folds:

1. The model is initialized from scratch with random weights.  
2. The training subset is used for model learning.  
3. After every epoch, the model is evaluated on the corresponding validation subset.  
4. Training and validation metrics (loss and ROC-AUC) are computed at each epoch.  
5. The model achieving the highest validation ROC-AUC during training is saved as the best model for that fold.

All metrics are stored per epoch and per fold for later analysis.  
After all five folds are completed:

- The **best validation AUC** from each fold is collected.  
- The **mean and standard deviation** of these best AUC values are reported as the baseline’s final performance.
