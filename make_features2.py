from tqdm import tqdm

import pandas as pd
import numpy as np

import os
import os.path

import warnings; warnings.filterwarnings('ignore')

PATH='individual_features'

# Read the extracted features
import glob
features = []
for file in tqdm(glob.glob(os.path.join(PATH, 'features_tpehg*.csv'))):
    features.append(pd.read_csv(file, index_col=0))
features = pd.concat(features)

clin_features = ['id', 'channel', 'RecID', 'Gestation', 'Rectime', 
                 'Age', 'Parity', 'Abortions', 'Weight', 'Hypertension', 
                 'Diabetes', 'Placental_position', 'Bleeding_first_trimester', 
                 'Bleeding_second_trimester', 'Funneling', 'Smoker']

# Create some extra columns
features['Gestation'] = features['Gestation'].astype(float)
features['Rectime'] = features['Rectime'].astype(float)
features['TimeToBirth'] = features['Gestation'] - features['Rectime']
features['Term'] = features['Gestation'] >= 37

# Create a feature matrix by concatenating the features of the three channels per sample
features[['Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight']] = features[['Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight']].replace(to_replace='None', value=np.NaN)

ids = set(features['id'])
channels = set(features['channel'])
joined_features = []
for _id in tqdm(ids):
    features_id = []
    features_filtered = features[features['id'] == _id]
    for channel in channels:
        channel_features = features_filtered[features_filtered['channel'] == channel]
        col_map = {}
        for col in channel_features:
            if col not in clin_features:
                col_map[col] = '{}_ch{}'.format(col, channel)
        channel_features = channel_features.rename(columns=col_map)
        features_id.append(channel_features)
    features_id = pd.concat(features_id, axis=1)
    joined_features.append(features_id)
joined_features = pd.concat(joined_features)
joined_features = joined_features.loc[:,~joined_features.columns.duplicated()]

joined_features = pd.get_dummies(joined_features, columns=['Hypertension', 'Diabetes', 'Placental_position', 'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker'])
for col in ['Gestation', 'Rectime', 'Age', 'Parity', 'Abortions', 'Weight']:
    joined_features[col] = joined_features[col].fillna(joined_features[col].mean())
    
for col in joined_features.columns[joined_features.isnull().sum() > 0]:
    joined_features[col] = joined_features[col].fillna(joined_features[col].mean())
    
ttb = joined_features['TimeToBirth_ch1']
feature_matrix = joined_features.drop(['TimeToBirth_ch3', 'TimeToBirth_ch2', 'TimeToBirth_ch1', 
                                       'RecID', 'channel'], axis=1) # , 'id'

X= feature_matrix.reset_index(drop=True)
khan_features = [
	'FeaturesJager_fmed_ch1', 'FeaturesJager_max_lyap_ch1', 
	'FeaturesJager_sampen_ch1', 'FeaturesJager_fmed_ch2', 
	'FeaturesJager_max_lyap_ch2', 'FeaturesJager_sampen_ch2',
	'FeaturesJager_fmed_ch3', 'FeaturesJager_max_lyap_ch3', 
	'FeaturesJager_sampen_ch3',
]
X= X[[c for c in X.columns if c in khan_features or ('FeaturesAcharya' in c and 'SampleEntropy' in c)]]
y= feature_matrix['Rectime'] + ttb >= 37


X.to_csv('raw_features.csv')
y.to_csv('target.csv', index=False)