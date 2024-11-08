from smote_variants import ADASYN
from smote_variants.classifiers import OversamplingClassifier

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import json

from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin, TransformerMixin

class AhmedStudy(ClassifierMixin):
    def __init__(self, grid=True, preprocessing=StandardScaler(), random_state= 5):
        self.grid= grid
        self.preprocessing= preprocessing
        self.random_state= random_state

    def fit(self, X, y):
        base_classifier= SVC(probability=True, random_state=self.random_state)
        grid_search_params= {'C': [10**i for i in range(-4, 5)]}

        oversampler = ('smote_variants', 'ADASYN', {'random_state':self.random_state})
        classifier = ('sklearn.svm', 'SVC', {'probability':True, 'random_state':self.random_state})
        classifier= classifier if not self.grid else ('sklearn.model_selection', 'GridSearchCV', {'estimator':base_classifier, 'param_grid':grid_search_params, 'scoring':'roc_auc'})
        classifier= OversamplingClassifier(oversampler, classifier)
        self.pipeline= classifier if not self.preprocessing else Pipeline([('preprocessing', self.preprocessing), ('classifier', classifier)])
        self.pipeline.fit(X, y)

        return self

    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

def evaluate(pipeline, X, y, validator):
    preds= np.zeros((len(X), 3))
    for fold_idx, (train_idx, test_idx) in enumerate(validator.split(X, y)):
        X_train, X_test= X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test= y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train.values, y_train.values)

        preds[test_idx, 0]= fold_idx
        preds[test_idx, 1]= y_test
        preds[test_idx, 2]= pipeline.predict_proba(X_test.values)[:,1]

    preds= pd.DataFrame(preds, columns=['fold', 'label', 'prediction'])

    return preds

def study_ahmed(features, target, preprocessing=StandardScaler(), grid=True, random_seed=42, output_file='idowu.json'):
    results= {}
    base_classifier= SVC(probability=True, random_state=random_seed)
    grid_search_params= {'C': [10**i for i in range(-4, 5)]}

    np.random.seed(random_seed)

    # without oversampling
    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=5, random_state= random_seed, shuffle=True)

    preds= evaluate(pipeline, features, target, validator)
    results['without_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['without_oversampling_details']= preds.to_dict()

    print('without oversampling: ', results['without_oversampling_auc'])

    # with correct oversampling
    oversampler = ('smote_variants', 'ADASYN', {'random_state':random_seed})
    classifier = ('sklearn.svm', 'SVC', {'probability':True, 'random_state':random_seed})
    classifier= classifier if not grid else ('sklearn.model_selection', 'GridSearchCV', {'estimator':base_classifier, 'param_grid':grid_search_params, 'scoring':'roc_auc'})
    classifier= OversamplingClassifier(oversampler, classifier)
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=5, random_state= random_seed, shuffle=True)

    preds= evaluate(classifier, features, target, validator)
    results['with_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['with_oversampling_details']= preds.to_dict()
    print('with oversampling: ', results['with_oversampling_auc'])

    # in-sample evaluation
    classifier= base_classifier
    preds= classifier.fit(features.values, target.values).predict_proba(features.values)[:,1]
    preds= pd.DataFrame({'fold': 0, 'label': target.values, 'prediction': preds})
    results['in_sample_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['in_sample_details']= preds.to_dict()
    print('in sample: ', results['in_sample_auc'])

    # with incorrect oversampling
    X, y= ADASYN(random_state=random_seed).sample(features.values, target.values)
    X= pd.DataFrame(X, columns=features.columns)
    y= pd.Series(y)

    classifier= base_classifier if not grid else GridSearchCV(base_classifier, grid_search_params, scoring='roc_auc')
    pipeline= classifier if not preprocessing else Pipeline([('preprocessing', preprocessing), ('classifier', classifier)])
    validator= StratifiedKFold(n_splits=5, random_state=random_seed, shuffle=True)

    preds= evaluate(pipeline, X, y, validator)
    results['incorrect_oversampling_auc']= roc_auc_score(preds['label'].values, preds['prediction'].values)
    results['incorrect_oversampling_details']= preds.to_dict()
    print('incorrect oversampling: ', results['incorrect_oversampling_auc'])


    json.dump(results, open(output_file, 'w'))
    return results