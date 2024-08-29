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
from sklearn.metrics import accuracy_score, recall_score
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

In this section, we will train and evaluate the SVM-FG model without the data leakage problem - i.e. with a correct evaluation on new EHG samples, which were not used to oversample training set.

:::

::: {.cell .markdown}

We will do this in three parts:

* **Oversampling training and test sets separately**: First, we will do the oversampling inside our KFold CV loop - i.e. oversample training data and test data separately to get balanced classes in each set. We will train the model on the training set and evaluate on test set. We will see that the accuracy due to this procedure is somewhat less than in the earlier example, where we oversampled before dividing into training and test.

* **Oversampling training set and evaluating on unprocessed test set**: The approach above addresses the data leakage due to oversampling on the entire data set together. However, it is still not a realistic view of how our model will perform in real usage, when the prevalence of pre-term birth is much less. In the next part, we will evaluate our model using unprocessed (not oversampled) test set, to get a better estimate of its real performance.

* **Using pipelines**: When training a machine learning model with preprocessing steps, it can be helpful to use a [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to cross validate the model fitting and preprocessing together. Since we use a preprocessing step from the `imbalanced-learn` package, we use their [Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html) implementation.


:::

::: {.cell .markdown}
### Oversampling training and test sets separately

In this approach, we create a `StratifiedKFold` instance to perform 5-fold cross validation. We then define a hyperparameter grid for `GridSearchCV` and initialize numpy arrays to store metrics obtained for each fold. The process for each fold is as follows:

- Oversample the training set using ADASYN [4] to address class imbalance.
- Use `GridSearchCV` to find optimal hyperparameter values.
- Plot validation accuracy for various combinations of gamma and C parameters obtained during `GridSearchCV`.
- Oversample the test set.
- Evaluate the optimized classifier on the oversampled test set.
- Store the resulting metrics in the corresponding arrays.

Finally, we report the mean values for test accuracy, sensitivity, specificity, and error rate across all folds.

_Note: `StratifiedKFold` is used to obtain balanced folds, ensuring equal representation of minority samples across folds_

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
metrics = {'accuracy': np.zeros(5), 'error': np.zeros(5), 'specificity': np.zeros(5), 'sensitivity': np.zeros(5)}

# Create figure to plot heatmaps
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(36, 6))

# Initialize ADASYN oversampler
oversampler = imblearn.over_sampling.ADASYN(n_neighbors=5, random_state=15)

# Loop through the folds of the cross-validation
for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):

    # Split the data into training and test sets
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    
    # Oversample the training set
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

    # Train the model on the oversampled training set
    clf.fit(X_train_oversampled, y_train_oversampled)

    # Plot the grid search results
    scores = clf.cv_results_["mean_test_score"].reshape(C_range.shape[0], gamma_range.shape[0])
    im=axes[fold].imshow(scores, interpolation="nearest", cmap='viridis', vmin=0.0, vmax=1.0)
    axes[fold].set_xlabel("gamma")
    axes[fold].set_ylabel("C")
    axes[fold].set_xticks(np.arange(gamma_range.shape[0]), labels=gamma_range, rotation=45)
    axes[fold].set_yticks(np.arange(gamma_range.shape[0]), labels=C_range)
    axes[fold].set_title(f"Validation accuracy fold {fold + 1}")

    # Oversample test set
    X_test_oversampled, y_test_oversampled = oversampler.fit_resample(X_test, y_test)

    # Evaluate the model on the test set
    y_pred_oversampled = clf.predict(X_test_oversampled)

    # Compute metrics
    metrics['accuracy'][fold] = accuracy_score(y_test_oversampled, y_pred_oversampled)
    metrics['error'][fold] = 1 - accuracy_score(y_test_oversampled, y_pred_oversampled)
    metrics['sensitivity'][fold] = recall_score(y_test_oversampled, y_pred_oversampled)
    metrics['specificity'][fold] = recall_score(y_test_oversampled, y_pred_oversampled, pos_label=0)

# Display the heatmaps
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
### Oversampling training set and evaluating on unprocessed test set

This approach follows a similar methodology to the previous one, with a modification: **we omit oversampling the test set**. By preserving the original distribution of the test set, we maintain its representation of real-world data, thus providing a more accurate evaluation of the model's practical performance.

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
metrics = {'accuracy': np.zeros(5), 'error': np.zeros(5), 'specificity': np.zeros(5), 'sensitivity': np.zeros(5)}

# Create figure to plot heatmaps
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(36, 6))

# Initialize ADASYN oversampler
oversampler = imblearn.over_sampling.ADASYN(n_neighbors=5, random_state=15)

# Loop through the folds of the cross-validation
for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):

    # Split the data into training and test sets
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    
    # Oversample the training set
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

    # Train the model on the oversampled training set
    clf.fit(X_train_oversampled, y_train_oversampled)

    # Plot the grid search results
    scores = clf.cv_results_["mean_test_score"].reshape(C_range.shape[0], gamma_range.shape[0])
    im=axes[fold].imshow(scores, interpolation="nearest", cmap='viridis', vmin=0.0, vmax=1.0)
    axes[fold].set_xlabel("gamma")
    axes[fold].set_ylabel("C")
    axes[fold].set_xticks(np.arange(gamma_range.shape[0]), labels=gamma_range, rotation=45)
    axes[fold].set_yticks(np.arange(gamma_range.shape[0]), labels=C_range)
    axes[fold].set_title(f"Validation accuracy fold {fold + 1}")

    # Evaluate the model on the test set
    y_pred = clf.predict(X_test)

    # Compute metrics
    metrics['accuracy'][fold] = accuracy_score(y_test, y_pred)
    metrics['error'][fold] = 1 - accuracy_score(y_test, y_pred)
    metrics['sensitivity'][fold] = recall_score(y_test, y_pred)
    metrics['specificity'][fold] = recall_score(y_test, y_pred, pos_label=0)

# Display the heatmaps
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

In our final approach, we demonstrate that the above implementation can be achieved using `Pipeline` and `GridSearchCV`.

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

# Define GridSearchCV
clf = GridSearchCV(pipe, param_grid, cv=10, scoring='accuracy')

# Perform cross-validation
cv_results = cross_validate(clf, X.to_numpy(), y.to_numpy(), scoring='accuracy', cv=5)

# Average performance on the test set
print("Performance on test set:")
print(cv_results['test_score'].mean())
# Standard error of the performance metrics
print("Standard error:")
print(cv_results['test_score'].std()/cv_results['test_score'].shape[0])
```
:::

::: {.cell .markdown}
## Discussion

| Metric        | Original | Reproduced With Data Leakage | Reproduced Without Data Leakage (Oversampled test set) | Reproduced Without Data Leakage (Unprocessed test set) |
|:-------------:|:--------:|:----------------------------:|:--------------------:|:---------------:|
| Accuracy      | 95.5     | 97.12 ± 0.74                 | 51.84 ± 2.75         | 84.56 ± 1.62    | 
| Error         | 4.48     | 2.88 ± 0.74                  | 48.16 ± 2.75         | 15.43 ± 1.62    |
| Specificity   | 97.13    | 99.62 ± 0.38                 | 6.2 ± 5.24           | 0.00 ± 0.00     |
| Sensitivity   | 93.51    | 94.62 ± 1.41                 | 96.92 ± 1.15         | 96.92 ± 1.88    |

By comparing the results from 'Reproduced With Data Leakage' and 'Reproduced Without Data Leakage (Oversampled test set)', we can verify that data leakage led to overly optimistic estimates of model performance. While we oversampled the test set in the latter scenario to provide a fair comparison, this is not recommended in practice for the following reasons:

- It changes the distribution of the test data.
- The oversampled test set no longer represents real-world data.

To provide a realistic view of the model's performance, we have also reported metrics obtained using correct preprocessing and evaluation procedures without modifying the test set. These results reflect the model's performance on unseen data.

:::

::: {.cell .markdown}
## References

[1]: M. U. Khan, S. Aziz, S. Ibraheem, A. Butt and H. Shahid, "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures," 2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), Vancouver, BC, Canada, 2019, pp. 0899-0905, doi: 10.1109/IEMCON.2019.893629

[2]: Fele-Žorž, G., Kavšek, G., Novak-Antolič, Ž. et al. A comparison of various linear and non-linear signal processing techniques to separate uterine EMG records of term and pre-term delivery groups. Med Biol Eng Comput 46, 911–922 (2008). https://doi.org/10.1007/s11517-008-0350-y

[3]: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

[4]: Haibo He, Yang Bai, E. A. Garcia and Shutao Li, "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 1322-1328, doi: 10.1109/IJCNN.2008.4633969. keywords: {Classification algorithms;Decision trees;Algorithm design and analysis;Training data;Machine learning;Accuracy;Machine learning algorithms}

:::