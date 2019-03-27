# FIV, Jan 2019
print('---------------------------------------------------------------------------------------')
print('Statistical univariate analysis and correlations')
print('Usage: > python statistics.py configuration.txt')
print('FIV, Jan 2019, http://cn.tuwien.ac.at')
print('---------------------------------------------------------------------------------------')

import csv
import fileinput
import pandas as pd
import numpy as np
import os

config_files={'training_data(in)':"AGM_training.csv", 'test_data(in)':"AGM_testing.csv", 'datasetID(in)':"Example", 'feat_vec(in)':"default", 'folder(out)':"Example/results"}
"""
# Read configuration file
for line in fileinput.input():
	name,val=line.split(":")
	if (name in config_files):
		config_files[name]=val.rstrip()
"""
# read datasets
print "Loading data..."
print "Training set:\t", config_files['training_data(in)']
training = pd.read_csv(config_files['training_data(in)']).fillna(0)
print "Test set:\t", config_files['test_data(in)']
test = pd.read_csv(config_files['test_data(in)']).fillna(0)
feat_vec = config_files['datasetID(in)']
datasetID = config_files['feat_vec(in)']
print "Dataset:\t", config_files['datasetID(in)']
print "Feature vector:\t", config_files['feat_vec(in)']

rate=0.1
print "Calculating Pearson's correlation based on a data sample (rate)..."
training.describe().to_csv(config_files['folder(out)']+"/"+"training_stats.csv")
test.describe().to_csv(config_files['folder(out)']+"/"+"test_stats.csv")
sample=training.sample(frac=rate, replace=True, random_state=1)
sample.corr(method='spearman').to_csv(config_files['folder(out)']+"/"+"training_corrs.csv")
sample=test.sample(frac=rate, replace=True, random_state=1)
sample.corr(method='spearman').to_csv(config_files['folder(out)']+"/"+"test_corrs.csv")

import matplotlib.pyplot as plt
    
feat_ind=list(training.select_dtypes(include=[np.float64,np.int64]).columns)
for x in range(0, len(feat_ind)-1):
    fig, ax = plt.subplots()
    training.hist(column=feat_ind[x], bins=100, ax=ax)  # arguments are passed to np.histogram
    fig.savefig(os.path.join(config_files['folder(out)'], "numerical", feat_ind[x]+"_training.png"))

feat_ind=list(training.select_dtypes(include=[np.object]).columns)
for x in range(0, len(feat_ind)-1):
    fig, ax = plt.subplots()    
    training[feat_ind[x]].value_counts().plot(ax=ax, kind='bar')
    fig.savefig(os.path.join(config_files['folder(out)'], "categorical", feat_ind[x]+"_training.png"))
