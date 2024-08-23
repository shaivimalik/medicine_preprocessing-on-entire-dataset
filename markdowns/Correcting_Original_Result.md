::: {.cell .markdown}

# "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures" Without Data Leakage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/Correcting_Original_Result.ipynb)

:::

::: {.cell .markdown}
## Introduction

In the preceding notebooks, we demonstrated the impact of data leakage on a model's performance on the test set and real-world data. In this notebook, we will reproduce the results published in **Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures**[^1] without the data leakage error. Our goal here is to demonstrate the correct way of preprocessing the dataset and discuss the changes in the reported metrics upon rectification of the error.

### Objectives

- Implement the described techniques and train SVM without data leakage errors.
- Analyze and compare our results with those published in the paper and obtained in the previous notebook.

:::

::: {.cell .code}
```python
# Uncomment the following lines if running on Google Colab
#!git clone https://github.com/shaivimalik/medicine_preprocessing-on-entire-dataset.git
#!pip install -r requirements.txt
#%cd medicine_preprocessing-on-entire-dataset/notebooks
```
:::

::: {.cell .markdown}
## Retrieve the data & Generate Features

The **Term-Preterm EHG Database**[^2] is a collection of EHG signals obtained from 1997 to 2005 at the University Medical Centre Ljubljana, Department of Obstetrics and Gynecology. Electrohysterograms signatures are obtained by placing four electrodes on the abdomen of the mother. The TPEGH DB consists of EHG records obtained from 262 women who had full-term pregnancies and 38 whose pregnancies ended prematurely. Each record is composed of three channels, recorded from 4 electrodes. The differences in the electrical potentials of the electrodes were recorded, producing 3 channels. Each record consists of two files, a header file (.hea) containing information regarding the record and the data file (.dat) containing signal data[^3].

We'll begin by acquiring the TPEGH DB (Term-Preterm ElectroHysteroGram Database) and extracting relevant features for our model training. The following cell will:

- Clone the project repository
- Download the TPEGH DB dataset
- Install required dependencies

Note that the download may take some time depending on your internet connection speed.

:::

::: {.cell .code}
```python
!curl -o ../term-preterm-ehg-database-1.0.1.zip https://physionet.org/static/published-projects/tpehgdb/term-preterm-ehg-database-1.0.1.zip
!unzip ../term-preterm-ehg-database-1.0.1.zip -d ../
``` 

We will use Empirical Mode Decomposition to extract Intrinsic Mode Functions from raw EHG signatures. Then, we will compute Median frequency, Shannon energy, Log energy and Lyapunov exponent from IMF-1. These computed features will be used for training our model.

The code cell below automates this process. It creates a directory named `individual_features` to store the feature files for each signal. Then, it executes two Python scripts:

- `all_features.py`: generates the individual feature files for each signal.

- `process_feature_files.py`: combines the individual feature files into a single dataset containing features from all 298 EHG signatures.

Note that 2 EHG signals will be discarded due to their short recording lengths.

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, make_scorer
```
:::

::: {.cell .markdown}

We load feature matrix (features) and labels (y) from CSV files. The `head()` function displays the first few rows of each dataframe for a quick overview.

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

In this section, we will train and evaluate the SVM-FG model without the data leakage problem - i.e. with a correct evaluation on new EHG samples not used in training.


:::

::: {.cell .markdown}

We will do this in three parts:

* **Oversampling training and validation sets separately**: First, we will do the oversampling inside our KFold CV loop - i.e. oversample training data and validation data separately to get balanced classes in each set. We will train the model on the training set and evaluate on validation set. We will see that the accuracy due to this procedure is somewhat less than in the earlier example, where we oversampled before dividing into training and validation.

* **Oversampling training set and evaluating on unprocessed validation set**: The approach above addresses the data leakage due to oversampling on the entire data set together. However, it is still not a realistic view of how our model will perform in real usage, when the prevalence of pre-term birth is much less. In the next part, we will evaluate our model using unprocessed (not oversampled) validation set, to get a better estimate of its real performance.

* **Using pipelines**: When training a machine learning model with preprocessing steps, it can be helpful to use a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to cross validate the model fitting and preprocessing together. Since we use a preprocessing step from the `imbalanced-learn` package, we use their [Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html) implementation.


:::

::: {.cell .markdown}
### Oversampling training and validation sets separately

:::

::: {.cell .code}
```python
# Training model

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

# Define the parameter grid for GridSearchCV
gamma_range = np.logspace(start=-5, stop=5, num=11, base=10)
C_range = np.logspace(start=-5, stop=5, num=11, base=10)
param_grid = {'C': C_range, 'gamma': gamma_range}

# Create SVC and GridSearchCV
svc = SVC(kernel='rbf', random_state=15)
clf = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy')

# Dictionary to store performance metrics
metrics = {'accuracy': [], 'error': [], 'specificity': [], 'sensitivity': []}

# Create figure to plot heatmaps
fig, axes = plt.subplots(1, 5, figsize=(30, 12))

oversampler = imblearn.over_sampling.ADASYN(n_neighbors=5, random_state=15)

# Loop through the folds of the cross-validation
for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
    # Split the data into training and testing sets
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
    # Train the model on training set
    clf.fit(X_train_oversampled, y_train_oversampled)

    # Plot the grid search results
    scores = clf.cv_results_["mean_test_score"].reshape(C_range.shape[0], gamma_range.shape[0])
    im=axes[fold].imshow(scores, interpolation="nearest", cmap=plt.cm.hot)
    axes[fold].set_xlabel("gamma")
    axes[fold].set_ylabel("C")
    axes[fold].set_xticks(np.arange(gamma_range.shape[0]), labels=gamma_range, rotation=45)
    axes[fold].set_yticks(np.arange(gamma_range.shape[0]), labels=C_range)
    axes[fold].set_title(f"Validation accuracy fold {fold + 1}")

    # Oversample test set
    X_test_oversampled, y_test_oversampled = oversampler.fit_resample(X_test, y_test)
    # Evaluate the model on the testing set
    y_pred_oversampled = clf.predict(X_test_oversampled)
    # Compute metrics
    metrics['accuracy'].append(accuracy_score(y_test_oversampled, y_pred_oversampled))
    metrics['error'].append(1 - accuracy_score(y_test_oversampled, y_pred_oversampled))
    metrics['sensitivity'].append(recall_score(y_test_oversampled, y_pred_oversampled))
    metrics['specificity'].append(recall_score(y_test_oversampled, y_pred_oversampled, pos_label=0))
    
fig.show()

# Create a DataFrame from the performance metrics
metrics_df = pd.DataFrame(metrics)

# Average performance on the test set
print("Performance on test set:")
print(metrics_df.mean())
# Standard error of the performance metrics
print("Standard error:")
print(metrics_df.std() / np.sqrt(kfold.get_n_splits()))
```
:::

::: {.cell .markdown}
### Oversampling training set and evaluating on unprocessed validation set

:::

::: {.cell .code}
```python
# Training model

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

# Define the parameter grid for GridSearchCV
gamma_range = np.logspace(start=-5, stop=5, num=11, base=10)
C_range = np.logspace(start=-5, stop=5, num=11, base=10)
param_grid = {'C': C_range, 'gamma': gamma_range}

# Create SVC and GridSearchCV
svc = SVC(kernel='rbf', random_state=15)
clf = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy')

# Dictionary to store performance metrics
metrics = {'accuracy': [], 'error': [], 'specificity': [], 'sensitivity': []}

# Create figure to plot heatmaps
fig, axes = plt.subplots(1, 5, figsize=(30, 12))

oversampler = imblearn.over_sampling.ADASYN(n_neighbors=5, random_state=15)

# Loop through the folds of the cross-validation
for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
    # Split the data into training and testing sets
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
    # Train the model on training set
    clf.fit(X_train_oversampled, y_train_oversampled)

    # Plot the grid search results
    scores = clf.cv_results_["mean_test_score"].reshape(C_range.shape[0], gamma_range.shape[0])
    im=axes[fold].imshow(scores, interpolation="nearest", cmap=plt.cm.hot)
    axes[fold].set_xlabel("gamma")
    axes[fold].set_ylabel("C")
    axes[fold].set_xticks(np.arange(gamma_range.shape[0]), labels=gamma_range, rotation=45)
    axes[fold].set_yticks(np.arange(gamma_range.shape[0]), labels=C_range)
    axes[fold].set_title(f"Validation accuracy fold {fold + 1}")

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)
    # Compute metrics
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['error'].append(1 - accuracy_score(y_test, y_pred))
    metrics['sensitivity'].append(recall_score(y_test, y_pred))
    metrics['specificity'].append(recall_score(y_test, y_pred, pos_label=0))
    
fig.show()

# Create a DataFrame from the performance metrics
metrics_df = pd.DataFrame(metrics)

# Average performance on the test set
print("Performance on test set:")
print(metrics_df.mean())
# Standard error of the performance metrics
print("Standard error:")
print(metrics_df.std() / np.sqrt(kfold.get_n_splits()))
```
:::

::: {.cell .markdown}
### Using pipelines

:::

::: {.cell .code}
```python
# Define specificity and sensitivity scoring functions
def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)
def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)

# Create scorers using make_scorer
specificity = make_scorer(specificity_score)
sensitivity = make_scorer(sensitivity_score)

fig, axes = plt.subplots(1, 5, figsize=(30, 12))

# Define the parameter grid for GridSearchCV
gamma_range = np.logspace(start=-5, stop=5, num=11, base=10)
C_range = np.logspace(start=-5, stop=5, num=11, base=10)
parameters={'SVM__C': C_range, 'SVM__gamma': gamma_range}

# Define scoring metrics for grid search
scoring = {'accuracy':'accuracy','sensitivity':sensitivity,'specificity':specificity}

# Define the pipeline
model = imblearn.pipeline.Pipeline([
        ('ADASYN', imblearn.over_sampling.ADASYN(random_state=5)),
        ('SVM', SVC(kernel='rbf', random_state=5))
    ])

# Define GridSearchCV
clf = GridSearchCV(model, parameters, cv=10, scoring=scoring, refit='accuracy')

# Loop through the folds of the cross-validation
for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
    # Split the data into training and testing sets
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]

    # Perform grid search
    clf.fit(X_train, y_train)

    # Plot the grid search results
    scores = clf.cv_results_["mean_test_accuracy"].reshape(C_range.shape[0], gamma_range.shape[0])
    im=axes[fold].imshow(scores, interpolation="nearest", cmap=plt.cm.hot)
    axes[fold].set_xlabel("gamma")
    axes[fold].set_ylabel("C")
    axes[fold].set_xticks(np.arange(gamma_range.shape[0]), labels=gamma_range, rotation=45)
    axes[fold].set_yticks(np.arange(gamma_range.shape[0]), labels=C_range)
    axes[fold].set_title(f"Validation accuracy fold {fold + 1}")

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)
    # Compute metrics
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['error'].append(1 - accuracy_score(y_test, y_pred))
    metrics['sensitivity'].append(recall_score(y_test, y_pred))
    metrics['specificity'].append(recall_score(y_test, y_pred, pos_label=0))
    
fig.show()

# Create a DataFrame from the performance metrics
metrics_df = pd.DataFrame(metrics)

# Average performance on the test set
print("Performance on test set:")
print(metrics_df.mean())
# Standard error of the performance metrics
print("Standard error:")
print(metrics_df.std() / np.sqrt(kfold.get_n_splits()))
```
:::

::: {.cell .markdown}
## Discussion

| Metric        | Original | Reproduced With Data Leakage | Reproduced Without Data Leakage (Oversampled test set) | Reproduced Without Data Leakage (Unprocessed test set) | Reproduced Without Data Leakage (Using pipeline) |
|:-------------:|:--------:|:----------------------------:|:--------------------:|:---------------:|:-------------:|
| Accuracy      | 95.5     | 97.12 ± 0.74                 | 49.71 ± 1.59         | 84.56 ± 1.62    | 85.91 ± 1.28
| Error         | 4.48     | 2.88 ± 0.74                  | 50.29 ± 1.59         | 15.43 ± 1.62    | 14.09 ± 1.28
| Specificity   | 97.13    | 99.62 ± 0.38                 | 2.99 ± 2.08          | 0.00 ± 0.00     | 0.00 ± 0.00
| Sensitivity   | 93.51    | 94.62 ± 1.41                 | 96.92 ± 1.88         | 96.92 ± 1.88    | 98.46 ± 1.45

:::

[^1]: M. U. Khan, S. Aziz, S. Ibraheem, A. Butt and H. Shahid, "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures," 2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), Vancouver, BC, Canada, 2019, pp. 0899-0905, doi: 10.1109/IEMCON.2019.893629

[^2]: Fele-Žorž, G., Kavšek, G., Novak-Antolič, Ž. et al. A comparison of various linear and non-linear signal processing techniques to separate uterine EMG records of term and pre-term delivery groups. Med Biol Eng Comput 46, 911–922 (2008). https://doi.org/10.1007/s11517-008-0350-y

[^3]: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

[^4]: Haibo He, Yang Bai, E. A. Garcia and Shutao Li, "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 1322-1328, doi: 10.1109/IJCNN.2008.4633969. keywords: {Classification algorithms;Decision trees;Algorithm design and analysis;Training data;Machine learning;Accuracy;Machine learning algorithms}