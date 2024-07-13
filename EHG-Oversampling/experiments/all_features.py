import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append('EHG-Oversampling') #add path to ehg-oversampling

from ehgfeatures.features import FeaturesAllEHG, FeatureGroup, FeaturesAcharya, FeaturesJager, FeaturesSubramaniam, FeaturesJanjarasjitt, FeaturesHosseinzahde, FeaturesSadiAhmed, FeaturesFergus, FeaturesTSFRESH, FeaturesAhmed, FeaturesRen, FeaturesJagerLibensek

from ehgfeatures.signal_io import get_signals

import pandas as pd

import time

import os
import joblib
from joblib import Parallel, delayed

import argparse

N_JOBS= 6

parser = argparse.ArgumentParser(description='file directories')
parser.add_argument("to_data", help="Path to the 'tpehgdb' directory (within 'term-preterm-ehg-database-1.0.1')")
parser.add_argument("to_store", help="Path to the directory where individual CSV files will be saved")
parser.add_argument("--study", help="Study for which features are required", default="FeaturesAllEHG")

args = parser.parse_args()

done = set([x.split('.')[0].split('_')[1] for x in os.listdir(args.to_store) if 'features_tpehg' in x])
    
DATA_PATH= args.to_data
    
ids, signals, all_clin_names, all_clin_values = get_signals(DATA_PATH, n_signals=-1)

print(len(signals), len(all_clin_names), len(all_clin_values))

class Khan_Features(FeatureGroup):
    def __init__(self):
        super().__init__(features=[FeaturesAcharya(), FeaturesJager()])

feature_dict={"FeaturesAllEHG":FeaturesAllEHG, "FeaturesAcharya":FeaturesAcharya, "FeaturesJager":FeaturesJager, \
    "FeaturesSubramaniam":FeaturesSubramaniam ,"FeaturesJanjarasjitt":FeaturesJanjarasjitt, "FeaturesHosseinzahde": \
    FeaturesHosseinzahde, "FeaturesSadiAhmed":FeaturesSadiAhmed, "FeaturesFergus":FeaturesFergus, "FeaturesTSFRESH":\
    FeaturesTSFRESH, "FeaturesAhmed":FeaturesAhmed, "FeaturesRen":FeaturesRen, "FeaturesJagerLibensek":FeaturesJagerLibensek\
    ,"FeaturesKhan":Khan_Features}

def process_one_item(parameters):
	i, (_id, (signal_ch1, signal_ch2, signal_ch3), clin_names, clin_values)= parameters

	if _id in done:
		print('Skipping {}'.format(_id))
		return

	print(_id, i)

	fe = feature_dict[args.study]()
	results_ch1 = fe.extract(signal_ch1[3000:-3000])
	results_ch1['id'] = _id
	results_ch1['channel'] = 1
	for name, value in zip(clin_names, clin_values):
		results_ch1[name] = [value]

	results_ch2 = fe.extract(signal_ch2[3000:-3000])
	results_ch2['id'] = _id
	results_ch2['channel'] = 2
	for name, value in zip(clin_names, clin_values):
		results_ch2[name] = [value]

	results_ch3 = fe.extract(signal_ch3[3000:-3000])
	results_ch3['id'] = _id
	results_ch3['channel'] = 3
	for name, value in zip(clin_names, clin_values):
		results_ch3[name] = [value]

	df1 = pd.DataFrame.from_dict(results_ch1, orient='columns')
	df2 = pd.DataFrame.from_dict(results_ch2, orient='columns')
	df3 = pd.DataFrame.from_dict(results_ch3, orient='columns')

	df = pd.concat([df1, df2, df3])

	df.to_csv(os.path.join(args.to_store,'features_{}.csv'.format(_id)))
   
Parallel(n_jobs=N_JOBS)(delayed(process_one_item)(parameters) for parameters in enumerate(zip(ids, signals, all_clin_names, all_clin_values)))