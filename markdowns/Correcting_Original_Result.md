::: {.cell .markdown}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/Correcting_Original_Result.ipynb)

# "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures" Without Data Leakage

:::

::: {.cell .markdown}
## Introduction

In the preceding notebooks, we demonstrated the impact of data leakage on a model's performance on the test set and real-world data. In this notebook, we will reproduce the results published in **Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures** [1] without the data leakage error. Our goal here is to demonstrate the correct way of preprocessing the dataset and discuss the changes in the reported metrics upon rectification of the error.

### Objectives

- Implement the described techniques and train SVM without data leakage errors.
- Analyze and compare our results with those published in the paper and obtained in the previous notebook.

:::

::: {.cell .code}
```python
# Uncomment the following lines if running on Google Colab
#!git clone https://github.com/shaivimalik/medicine_preprocessing-on-entire-dataset.git
#!pip install -r medicine_preprocessing-on-entire-dataset/requirements.txt
#%cd medicine_preprocessing-on-entire-dataset/notebooks
```
:::

::: {.cell .markdown}
## Retrieve the data & Generate Features

The **Term-Preterm EHG Database** [2] is a collection of EHG signals obtained from 1997 to 2005 at the University Medical Centre Ljubljana, Department of Obstetrics and Gynecology. The TPEHG DB consists of EHG records obtained from 262 women who had full-term pregnancies and 38 whose pregnancies ended prematurely. Each record consists of two files, a header file (.hea) containing information regarding the record and the data file (.dat) containing signal data [3].

We'll begin by acquiring the TPEHG DB (Term-Preterm ElectroHysteroGram Database) and extracting relevant features for our model training. 

_Note: The download may take some time depending on your internet connection speed._

:::

::: {.cell .code}
```python
!curl -o ../term-preterm-ehg-database-1.0.1.zip https://physionet.org/static/published-projects/tpehgdb/term-preterm-ehg-database-1.0.1.zip
!unzip ../term-preterm-ehg-database-1.0.1.zip -d ../
``` 
:::

::: {.cell .markdown}

We will use Empirical Mode Decomposition to extract Intrinsic Mode Functions from raw EHG signatures. Then, we will compute Median frequency, Shannon energy, Log energy and Lyapunov exponent from IMF-1. These computed features will be used for training our model. 

The code cell below automates this process. It creates a directory named `individual_features` to store the feature files for each signal. Then, it executes two Python scripts:

- `all_features.py`: generates the individual feature files for each signal.

- `process_feature_files.py`: combines the individual feature files into a single dataset containing features from all 298 EHG signatures.

_Note: 2 EHG signals will be discarded due to their short recording lengths._

:::

::: {.cell .code}
```python
!mkdir ../individual_features
!python3 ../EHG-Oversampling/experiments/all_features.py ../term-preterm-ehg-database-1.0.1/tpehgdb ../individual_features --study FeaturesKhan
!python3 ../EHG-Oversampling/experiments/process_feature_files.py ../individual_features ../
```
:::

::: {.cell .markdown}
## Loading the features

In this section, we will load the dataset from the CSV files created in the previous step. 

We start by importing the required modules.

:::

::: {.cell .code}
```python
import os
import imblearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import RocCurveDisplay, accuracy_score, recall_score, f1_score, recall_score, precision_score, balanced_accuracy_score, make_scorer, auc
```
:::

::: {.cell .markdown}

We load feature matrix (`features`) and labels (`y`) from CSV files. The `head()` function displays the first few rows of each dataframe for a quick overview.

:::

::: {.cell .code}
```python
#Loading feature vectors
features=pd.read_csv(os.path.join('..','raw_features.csv'))
features.head()
```
:::

::: {.cell .code}
```python
#Loading labels
y=pd.read_csv(os.path.join('..','target.csv'))
y.head()
```
:::

::: {.cell .code}
```python
# Extracting features required for our study
khan_features = [
    'FeaturesJager_fmed_ch1', 'FeaturesJager_max_lyap_ch1',
    'FeaturesJager_sampen_ch1', 'FeaturesJager_fmed_ch2',
    'FeaturesJager_max_lyap_ch2', 'FeaturesJager_sampen_ch2',
    'FeaturesJager_fmed_ch3', 'FeaturesJager_max_lyap_ch3',
    'FeaturesJager_sampen_ch3',
 ]
generic_features=[ c for c in features.columns if 'FeaturesAcharya' in c and 'SampleEntropy' in c ]

# Extract the relevant features for the study
X = features[khan_features + generic_features]

# Display summary information about the selected features
X.info()
```
:::

::: {.cell .markdown}
## SVM Classifier Training and Evaluation without Data Leakage

In this section, we will train and evaluate the SVM-FG model without the data leakage error - i.e. with a correct evaluation on new EHG samples, which were not used to oversample training set.

:::

::: {.cell .markdown}

We will do this in three parts:

* **Oversampling training and test sets separately**: First, we will do the oversampling inside our KFold CV loop - i.e. oversample training data and test data separately to get balanced classes in each set. We will train the model on the training set and evaluate on test set. We will see that the accuracy due to this procedure is somewhat less than in the earlier example, where we oversampled before dividing into training and test.

* **Oversampling training set and evaluating on unprocessed test set**: The approach above addresses the data leakage due to oversampling on the entire data set together. However, it is still not a realistic view of how our model will perform in real usage, when the prevalence of pre-term birth is much less. In this part, we will evaluate our model using unprocessed (not oversampled) test set, to get a better estimate of its real performance.

* **Using pipelines**: When training a machine learning model with preprocessing steps, it can be helpful to use a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to cross validate the model fitting and preprocessing together. Since we use a preprocessing step from the `imbalanced-learn` package, we use their [Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html) implementation.

_Note: We use 3-fold cross-validation instead of 10-fold to ensure sufficient minority samples in each fold._

:::

::: {.cell .markdown}
### Oversampling training and test sets separately

In this approach, we create a `StratifiedKFold` instance to perform 3-fold cross validation. We then define a hyperparameter grid for grid search and initialize numpy arrays to store metrics obtained for each fold. The process for each fold is as follows:

- Oversample the training set using ADASYN [4] to address class imbalance.
- Use grid search to find optimal hyperparameter values.
- Plot validation accuracy for various combinations of gamma and C parameters obtained during grid search.
- Oversample the test set.
- Evaluate the optimized classifier on the oversampled test set.
- Store the resulting metrics in the corresponding arrays.

Finally, we report the mean values for test accuracy, error, sensitivity, specificity, precision, f1-score and negative predictive value across all folds. We also present ROC curves for each fold. 

_Note: `StratifiedKFold` is used to obtain balanced folds, ensuring equal representation of minority samples across folds_

:::

::: {.cell .code}
```python
# Training model

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=15)

# Define the parameter grid for GridSearch
gamma_range = np.logspace(start=-5, stop=5, num=11, base=10)
C_range = np.logspace(start=-5, stop=5, num=11, base=10)

# Dictionary to store performance metrics
metrics_acc = {'test_accuracy': np.zeros(kfold.get_n_splits()), 'test_error': np.zeros(kfold.get_n_splits()), 'test_balanced_accuracy': np.zeros(kfold.get_n_splits()), 
                'test_specificity': np.zeros(kfold.get_n_splits()), 'test_sensitivity': np.zeros(kfold.get_n_splits()), 'test_precision': np.zeros(kfold.get_n_splits()), 
                'test_negative_predictive_value': np.zeros(kfold.get_n_splits()), 'test_f1_score (preterm birth)': np.zeros(kfold.get_n_splits())}

# Create figure to plot heatmaps
fig_heatmap, axes = plt.subplots(nrows=1, ncols=kfold.get_n_splits(), figsize=(36, 6))

# Initialize ADASYN oversampler
oversampler = imblearn.over_sampling.ADASYN(n_neighbors=5, random_state=15)

# Create figure to plot ROC curve
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 6))

# Loop through the folds of the cross-validation
for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):

    # Split the data into training and test sets
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]

    print("Grid Search Fold:",fold+1)

    # Initialize array to store mean val scores
    mean_val_score = np.zeros((C_range.shape[0], gamma_range.shape[0]))

    # Perform nested cross-validation for hyperparameter tuning
    for idx, (train_index_opt, val_index) in enumerate(kfold.split(X_train, y_train)):

        # Split training data
        X_train_opt, y_train_opt = X_train[train_index_opt], y_train[train_index_opt]
        X_val_opt, y_val_opt = X_train[val_index], y_train[val_index]

        # Apply oversampling to training and validation sets
        X_train_opt_oversampled, y_train_opt_oversampled = oversampler.fit_resample(X_train_opt, y_train_opt)
        X_val_opt_oversampled, y_val_opt_oversampled = oversampler.fit_resample(X_val_opt, y_val_opt)

        # Grid search over C and gamma parameters
        for i in range(C_range.shape[0]):
            for j in range(gamma_range.shape[0]):
                svc_opt = SVC(kernel='rbf', C=C_range[i], gamma=gamma_range[j], random_state=15)
                svc_opt.fit(X_train_opt_oversampled, y_train_opt_oversampled)
                y_pred_opt = svc_opt.predict(X_val_opt_oversampled)
                mean_val_score[i,j] += accuracy_score(y_val_opt_oversampled, y_pred_opt)

    # Calculate mean test score across all inner folds
    mean_val_score = mean_val_score/kfold.get_n_splits()
    
    # Find best hyperparameters
    C_index, gamma_index = np.unravel_index(np.argmax(mean_val_score, axis=None), mean_val_score.shape)
    print("C:",C_range[C_index])
    print("gamma:", gamma_range[gamma_index])
    print("Validation accuracy:", mean_val_score[C_index, gamma_index])

    # Plot heatmap
    im = axes[fold].imshow(mean_val_score, interpolation="nearest", cmap='Blues', vmin=0.0, vmax=1.0)
    axes[fold].set_ylabel("C")
    axes[fold].set_xlabel("gamma")
    axes[fold].set_xticks(np.arange(gamma_range.shape[0]), labels=gamma_range, rotation=45)
    axes[fold].set_yticks(np.arange(C_range.shape[0]), labels=C_range)
    axes[fold].set_title(f"Validation accuracy fold {fold + 1}")

    # Oversample train & test set
    X_test_oversampled, y_test_oversampled = oversampler.fit_resample(X_test, y_test)
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

    # Train model with optimal hyperparameters
    svm = SVC(kernel='rbf', C=C_range[C_index], gamma=gamma_range[gamma_index], random_state=15)
    svm.fit(X_train_oversampled, y_train_oversampled)

    # Evaluate the model on the test set
    y_pred_oversampled = svm.predict(X_test_oversampled)

    # Compute metrics
    metrics_acc['test_accuracy'][fold] = accuracy_score(y_test_oversampled, y_pred_oversampled)
    metrics_acc['test_error'][fold] = 1-accuracy_score(y_test_oversampled, y_pred_oversampled)
    metrics_acc['test_sensitivity'][fold] = recall_score(y_test_oversampled, y_pred_oversampled)
    metrics_acc['test_specificity'][fold] = recall_score(y_test_oversampled, y_pred_oversampled, pos_label=0)
    metrics_acc['test_precision'][fold] = precision_score(y_test_oversampled, y_pred_oversampled)
    metrics_acc['test_negative_predictive_value'][fold] = precision_score(y_test_oversampled, y_pred_oversampled, pos_label=0)
    metrics_acc['test_balanced_accuracy'][fold] = balanced_accuracy_score(y_test_oversampled, y_pred_oversampled)
    metrics_acc['test_f1_score (preterm birth)'][fold] = f1_score(y_test_oversampled, y_pred_oversampled, pos_label=0)

    # Plot ROC 
    viz = RocCurveDisplay.from_estimator(
        svm,
        X_test_oversampled,
        y_test_oversampled,
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(fold == kfold.get_n_splits() - 1),
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# Add colorbar to heatmap
plt.colorbar(im)
# Display the heatmaps
fig_heatmap.show()

# Display ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label 'term-birth')",
)
ax.legend(loc="lower right")
plt.show()

# Create a DataFrame from the performance metrics
metrics_df = pd.DataFrame(metrics_acc)

# Average performance on the test set
print("Average performance on test set:")
print(metrics_df.mean())
# Standard error of the performance metrics
print("Standard error:")
print(metrics_df.std() / np.sqrt(kfold.get_n_splits()))
```
:::

::: {.cell .markdown}

In the scenario with data leakage, we achieved an accuracy of 97% on a balanced test set but we can see that this was an overly optimistic evaluation because of data leakage. When we correct the data leakage by oversampling training, test and validation sets separately, we are only able to achieve 57% accuracy.

:::

::: {.cell .markdown}
### Oversampling training set and evaluating on unprocessed test set

This approach follows a similar methodology to the previous one, with a modification: **we omit oversampling the test set**. Evaluating on a balanced test set does not reflect true performance, as the model will be used in the real world, where the prevalence of preterm birth is less than 50%. By preserving the original distribution of the test set, we maintain its representation of real-world data, thus providing a more accurate evaluation of the model's practical performance.

_Note: Baseline model which predicts term birth for all samples would achieve an accuracy of 87% on the TPEHG DB._

:::

::: {.cell .code}
```python
# Training model

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=15)

# Define the parameter grid for GridSearch
gamma_range = np.logspace(start=-5, stop=5, num=11, base=10)
C_range = np.logspace(start=-5, stop=5, num=11, base=10)

# Dictionary to store performance metrics
metrics_acc_cor = {'test_accuracy': np.zeros(kfold.get_n_splits()), 'test_error': np.zeros(kfold.get_n_splits()), 'test_balanced_accuracy': np.zeros(kfold.get_n_splits()), 
                   'test_specificity': np.zeros(kfold.get_n_splits()), 'test_sensitivity': np.zeros(kfold.get_n_splits()), 'test_precision': np.zeros(kfold.get_n_splits()), 
                   'test_negative_predictive_value': np.zeros(kfold.get_n_splits()), 'test_f1_score (preterm birth)': np.zeros(kfold.get_n_splits())}

# Create figure to plot heatmaps
fig_heatmap, axes = plt.subplots(nrows=1, ncols=kfold.get_n_splits(), figsize=(36, 6))

# Initialize ADASYN oversampler
oversampler = imblearn.over_sampling.ADASYN(n_neighbors=5, random_state=15)

# Loop through the folds of the cross-validation
for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):

    # Split the data into training and test sets
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]

    print("Grid Search Fold:",fold+1)

    # Initialize array to store mean val scores
    mean_val_score = np.zeros((C_range.shape[0], gamma_range.shape[0]))

    # Perform nested cross-validation for hyperparameter tuning
    for idx, (train_index_opt, val_index) in enumerate(kfold.split(X_train, y_train)):

        # Split training data
        X_train_opt, y_train_opt = X_train[train_index_opt], y_train[train_index_opt]
        X_val_opt, y_val_opt = X_train[val_index], y_train[val_index]

        # Apply oversampling to training set
        X_train_opt_oversampled, y_train_opt_oversampled = oversampler.fit_resample(X_train_opt, y_train_opt)

        # Grid search over C and gamma parameters
        for i in range(C_range.shape[0]):
            for j in range(gamma_range.shape[0]):
                svc_opt = SVC(kernel='rbf', C=C_range[i], gamma=gamma_range[j], random_state=15)
                svc_opt.fit(X_train_opt_oversampled, y_train_opt_oversampled)
                y_pred_opt = svc_opt.predict(X_val_opt)
                mean_val_score[i,j] += accuracy_score(y_val_opt, y_pred_opt)

    # Calculate mean test score across all inner folds
    mean_val_score = mean_val_score/kfold.get_n_splits()
    
    # Find best hyperparameters
    C_index, gamma_index = np.unravel_index(np.argmax(mean_val_score, axis=None), mean_val_score.shape)
    print("C:",C_range[C_index])
    print("gamma:", gamma_range[gamma_index])
    print("Validation accuracy:", mean_val_score[C_index, gamma_index])

    # Plot heatmap
    im = axes[fold].imshow(mean_val_score, interpolation="nearest", cmap='Blues', vmin=0.0, vmax=1.0)
    axes[fold].set_ylabel("C")
    axes[fold].set_xlabel("gamma")
    axes[fold].set_xticks(np.arange(gamma_range.shape[0]), labels=gamma_range, rotation=45)
    axes[fold].set_yticks(np.arange(C_range.shape[0]), labels=C_range)
    axes[fold].set_title(f"Validation accuracy fold {fold + 1}")

    # Oversample train set
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

    # Train model with optimal hyperparameters
    svm = SVC(kernel='rbf', C=C_range[C_index], gamma=gamma_range[gamma_index], random_state=15)
    svm.fit(X_train_oversampled, y_train_oversampled)

    # Evaluate the model on the test set
    y_pred = svm.predict(X_test)

    # Compute metrics
    metrics_acc_cor['test_accuracy'][fold] = accuracy_score(y_test, y_pred)
    metrics_acc_cor['test_error'][fold] = 1-accuracy_score(y_test, y_pred)
    metrics_acc_cor['test_sensitivity'][fold] = recall_score(y_test, y_pred)
    metrics_acc_cor['test_specificity'][fold] = recall_score(y_test, y_pred, pos_label=0)
    metrics_acc_cor['test_precision'][fold] = precision_score(y_test, y_pred)
    metrics_acc_cor['test_negative_predictive_value'][fold] = precision_score(y_test, y_pred, pos_label=0, zero_division=0.0)
    metrics_acc_cor['test_balanced_accuracy'][fold] = balanced_accuracy_score(y_test, y_pred)
    metrics_acc_cor['test_f1_score (preterm birth)'][fold] = f1_score(y_test, y_pred, pos_label=0)

# Add colorbar to heatmap
plt.colorbar(im)
# Display the heatmaps
plt.show()

# Create a DataFrame from the performance metrics
metrics_df_cor = pd.DataFrame(metrics_acc_cor)

# Average performance on the test set
print("Average performance on test set:")
print(metrics_df_cor.mean())
# Standard error of the performance metrics
print("Standard error:")
print(metrics_df_cor.std() / np.sqrt(kfold.get_n_splits()))
```
:::

::: {.cell .markdown}
### Using pipelines

In our final approach, we demonstrate that the above implementation can be achieved using `Pipeline` and `GridSearchCV`.

:::

::: {.cell .code}
```python
eval_metrics = {"accuracy":"accuracy", "balanced_accuracy": "balanced_accuracy", "specificity": make_scorer(recall_score, pos_label=0), 
                "sensitivity": "recall", "precision": "precision", "negative_predictive_value": make_scorer(precision_score, pos_label=0, zero_division=0.0), 
                "f1_score (preterm birth)": make_scorer(f1_score, pos_label=0)}
```
:::

::: {.cell .code}
```python
# Define the parameter grid for GridSearchCV
gamma_range = np.logspace(start=-5, stop=5, num=11, base=10)
C_range = np.logspace(start=-5, stop=5, num=11, base=10)
param_grid = {'SVM__C': C_range, 'SVM__gamma': gamma_range}

# Define the pipeline
pipe = imblearn.pipeline.Pipeline([
        ('ADASYN', imblearn.over_sampling.ADASYN(random_state=5)),
        ('SVM', SVC(kernel='rbf', random_state=5))
    ])

# Define number of splits for StratifiedKFold
K=3

# Define GridSearchCV
clf = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=K, shuffle=True, random_state=15), scoring='accuracy')

# Perform cross-validation
cv_results_acc = cross_validate(clf, X.to_numpy(), y.to_numpy(), scoring=eval_metrics, cv=StratifiedKFold(n_splits=K, shuffle=True, random_state=15))

# Create a DataFrame from the performance metrics
cv_acc_df = pd.DataFrame(cv_results_acc)

# Average performance on the test set
print("Average performance on test set:")
print(cv_acc_df.mean())
# Standard error of the performance metrics
print("Standard error:")
print(cv_acc_df.std() / np.sqrt(K))
```
:::

::: {.cell .markdown}

Our model basically learns to predict term births for all samples. In this scenario with an imbalanced dataset, we might not want to optimise accuracy because we might care more about identifying women who are at risk of preterm birth.

Let's optimize for `balanced_accuracy` using `Pipeline` and `GridSearchCV`.

:::

::: {.cell .code}
```python
# Define the parameter grid for GridSearchCV
gamma_range = np.logspace(start=-5, stop=5, num=11, base=10)
C_range = np.logspace(start=-5, stop=5, num=11, base=10)
param_grid = {'SVM__C': C_range, 'SVM__gamma': gamma_range}

# Define the pipeline
pipe = imblearn.pipeline.Pipeline([
        ('ADASYN', imblearn.over_sampling.ADASYN(random_state=5)),
        ('SVM', SVC(kernel='rbf', random_state=5))
    ])

# Define number of splits for StratifiedKFold
K=3

# Define GridSearchCV
clf = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=K, shuffle=True, random_state=15), scoring='balanced_accuracy')

# Perform cross-validation
cv_results_balanced_acc = cross_validate(clf, X.to_numpy(), y.to_numpy(), scoring=eval_metrics, cv=StratifiedKFold(n_splits=K, shuffle=True, random_state=15))

# Create a DataFrame from the performance metrics
cv_bal_acc_df = pd.DataFrame(cv_results_balanced_acc)

# Average performance on the test set
print("Average performance on test set:")
print(cv_bal_acc_df.mean())
# Standard error of the performance metrics
print("Standard error:")
print(cv_bal_acc_df.std() / np.sqrt(K))
```
:::

::: {.cell .markdown}

Now let's optimize for specificity and see how it affects our results.

:::

::: {.cell .code}
```python
# Define the parameter grid for GridSearchCV
gamma_range = np.logspace(start=-5, stop=5, num=11, base=10)
C_range = np.logspace(start=-5, stop=5, num=11, base=10)
param_grid = {'SVM__C': C_range, 'SVM__gamma': gamma_range}

# Define the pipeline
pipe = imblearn.pipeline.Pipeline([
        ('ADASYN', imblearn.over_sampling.ADASYN(random_state=5)),
        ('SVM', SVC(kernel='rbf', random_state=5))
    ])

# Define number of splits for StratifiedKFold
K=3

# Define GridSearchCV
clf = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=K, shuffle=True, random_state=15), scoring=make_scorer(recall_score, pos_label=0))

# Perform cross-validation
cv_results_specificity = cross_validate(clf, X.to_numpy(), y.to_numpy(), scoring=eval_metrics, cv=StratifiedKFold(n_splits=K, shuffle=True, random_state=15))

# Create a DataFrame from the performance metrics
cv_specificity_df = pd.DataFrame(cv_results_specificity)

# Average performance on the test set
print("Average performance on test set:")
print(cv_specificity_df.mean())
# Standard error of the performance metrics
print("Standard error:")
print(cv_specificity_df.std() / np.sqrt(K))
```
:::

::: {.cell .markdown}

In this cell, we create a barplot to visualise our results.

:::

::: {.cell .code}
```python
results = [metrics_acc, cv_results_acc, cv_results_balanced_acc, cv_results_specificity]
labels = ['Grid Search Metric - Accuracy (Oversampled Test Set)', 'Grid Search Metric - Accuracy (Unprocessed Test Set)', 
              'Grid Search Metric - Balanced Accuracy (Unprocessed Test Set)', 'Grid Search Metric - Specificity (Unprocessed Test Set)'] 

# Get the keys 
keys = ['test_accuracy', 'test_balanced_accuracy', 'test_sensitivity', 'test_precision', 'test_specificity', 'test_f1_score (preterm birth)', 'test_negative_predictive_value']

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Set width of each bar and positions of the bars
bar_width = 0.2
r = np.arange(len(keys))

# Plot bars
for i, d in enumerate(results):
    values = [np.mean(d[key]) for key in keys]
    ax.bar(r + i*bar_width, values, width=bar_width, label=labels[i])

ax.set_xlabel('Metrics')
ax.set_ylabel('Mean Values')
ax.set_title('Comparison of Metric Values')
ax.set_xticks(r + bar_width * 1.5)
ax.set_xticklabels(keys, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize = 7)

# Display the plot
plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown}
## Discussion

| Metric        | Original | Reproduced With Data Leakage | Reproduced Without Data Leakage (Oversampled test set) | Reproduced Without Data Leakage (Unprocessed test set) |
|:-------------:|:--------:|:----------------------------:|:--------------------:|:---------------:|
| Accuracy      | 95.5     | 97.12 ± 0.74                 | 57.63 ± 3.56         | 87.25 ± 0.32    | 
| Error         | 4.48     | 2.88 ± 0.74                  | 42.37 ± 3.56         | 12.75 ± 0.32    |
| Specificity   | 97.13    | 99.62 ± 0.38                 | 35.00 ± 13.92        | 0.00 ± 0.00     |
| Sensitivity   | 93.51    | 94.62 ± 1.41                 | 79.68 ± 9.31         | 100.00 ± 0.00   |

By comparing the results from 'Reproduced With Data Leakage' and 'Reproduced Without Data Leakage (Oversampled test set)', we can verify that data leakage led to overly optimistic estimates of model performance. While we oversampled the test set in the latter scenario to provide a fair comparison, this is not recommended in practice for the following reasons:

- It changes the distribution of the test data.
- The oversampled test set no longer represents real-world data.

To provide a realistic view of the model's performance, we have also reported metrics obtained using correct preprocessing and evaluation procedures without modifying the test set. These results reflect the model's performance on unseen data.

| Metric        | Reproduced Without Data Leakage (Unprocessed test set) GridSearchCV scoring="accuracy" | Reproduced Without Data Leakage (Unprocessed test set) GridSearchCV scoring="balanced_accuracy" | Reproduced Without Data Leakage (Unprocessed test set) GridSearchCV scoring=make_scorer(recall_score, pos_label=0) (Specificity) |
|:-------------------------:|:------------:|:------------:|:-------------:|
| Accuracy                  | 87.25 ± 0.15 | 66.5 ± 4.88  | 27.5 ± 3.47   |   
| Balanced Accuracy         | 50.0 ± 0.0   | 48.25 ± 0.15 | 51.57 ± 0.5   |    
| Specificity               | 0.0 ± 0.0    | 23.72 ± 6.29 | 83.97 ± 3.79  | 
| Sensitivity               | 100.0 ± 0.0  | 72.79 ± 6.58 | 19.16 ± 4.61  | 
| Precision                 | 87.25 ± 0.15 | 86.64 ± 0.18 | 59.44 ± 14.01 | 
| Negative Predictive Value | 0.0 ± 0.0    | 7.57 ± 1.79  | 13.23 ± 0.03  | 
| F1-score (preterm birth)  | 0.0 ± 0.0    | 11.35 ± 2.73 | 22.81 ± 0.1   | 

Summary:

- With data leakage, we achieved high accuracy and were also able to detect preterm birth (high specificity) but this was not a true result.
- After correcting the data leakage, we achieve much lower accuracy (57.63%) and detect few preterm births (specificity=35%) on a balanced test set.
- When using a test set with a realistic preterm to term birth ratio:
    - if we optimise for accuracy, the model learns to predict term birth for all samples (accuracy = 87%, specificity=0%) which is not useful.
    - if we optimise for balanced accuracy, the model is able to predict some of the preterm birth (specificity=23.72%, accuracy=66.5%).
    - if we optimise for specificity, the model can predict more preterm birth although with lower accuracy overall (accuracy =27.5%, specificity=83.97%).


Fun Experiments: 

- **Classifier Exploration**: Test out Logistic Regression, Decision Trees, and Random Forests to see how each performs on our data.

- **Oversampling Techniques**: Try out SMOTE and RandomOverSampler to address the class imbalance. 

:::

::: {.cell .markdown}
## References

[1]: M. U. Khan, S. Aziz, S. Ibraheem, A. Butt and H. Shahid, "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures," 2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), Vancouver, BC, Canada, 2019, pp. 0899-0905, doi: 10.1109/IEMCON.2019.893629

[2]: Fele-Žorž, G., Kavšek, G., Novak-Antolič, Ž. et al. A comparison of various linear and non-linear signal processing techniques to separate uterine EMG records of term and pre-term delivery groups. Med Biol Eng Comput 46, 911–922 (2008). https://doi.org/10.1007/s11517-008-0350-y

[3]: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

[4]: Haibo He, Yang Bai, E. A. Garcia and Shutao Li, "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 1322-1328, doi: 10.1109/IJCNN.2008.4633969. keywords: {Classification algorithms;Decision trees;Algorithm design and analysis;Training data;Machine learning;Accuracy;Machine learning algorithms}

:::