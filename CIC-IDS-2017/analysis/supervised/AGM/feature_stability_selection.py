# FIV, Feb 2019
print('---------------------------------------------------------------------------------------')
print('Feature selection by usign stability selection with Logistic Regression')
print('Usage: > python feature_syability_selection.py configuration.txt')
print('FIV, Feb 2019, http://cn.tuwien.ac.at')
print('---------------------------------------------------------------------------------------')

import csv
import fileinput
import pandas as pd
import numpy as np

config_files={'training_data(in)':"AGM_testing.csv", 'test_data(in)':"AGM_training.csv", 'datasetID(in)':"Example", 'feat_vec(in)':"default", 'folder(out)':"Example/results"}
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
feat_vec = config_files['datasetID(in)']
datasetID = config_files['feat_vec(in)']
print "Dataset:\t", config_files['datasetID(in)']
print "Feature vector:\t", config_files['feat_vec(in)']

Xo = training.drop('Label', axis = 1)
yo = training['Label']
del training
names=list(Xo)
X=Xo
y=yo

# minmax scalling
from sklearn.preprocessing import MinMaxScaler
scalermm = MinMaxScaler()
scalermm.fit(Xo)
X = scalermm.transform(Xo)

fast=0 # bad for AGM
from sklearn.model_selection import train_test_split
if (fast):
    rate=0.1
    # Stratified sampling for reducing parameter search times 
    print('\nTaking subset for parameter search (10%) ')
    X, Xt, y, yt = train_test_split(X, y, test_size=1-rate, random_state=0, stratify=y)

from sklearn.model_selection import StratifiedShuffleSplit
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.linear_model import RandomizedLogisticRegression 
from sklearn.linear_model import LogisticRegression

c_range = np.logspace(-1, 2, 30)

print('\nRegularization Parameter C initial grid:')
print c_range
param_dist = dict(C=c_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=2018)
rnds = EvolutionaryAlgorithmSearchCV(
		estimator=LogisticRegression(random_state=0),
		params=param_dist,
		scoring="f1",
		cv=cv,
		verbose=1,
		population_size=50,
		gene_mutation_prob=0.10,
		gene_crossover_prob=0.5,
		tournament_size=3,
		generations_number=6,
		n_jobs=4)
rnds.fit(X, y)
best_C= rnds.best_estimator_.C

# apply best parameters
lr = RandomizedLogisticRegression(C=best_C, random_state=0, sample_fraction=0.75, n_resampling=200, selection_threshold=0.25)
lr.fit(X, y)

importances = lr.scores_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. %s - (feature %d) = %f" % (f + 1, names[indices[f]], indices[f], importances[indices[f]]))

evaluation=0
if (evaluation):
    rate=0.25 #decides how many features you want to remove
    Xo = Xo.values
    yo = yo.values
    print('\nTrain/test split (0.75/0.25) ')
    X, Xt, y, yt = train_test_split(Xo, yo, test_size=0.25, random_state=0, stratify=yo)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report,confusion_matrix
    dtree = DecisionTreeClassifier(min_samples_leaf=5,max_depth=10, random_state=0)
    dtree.fit(X,y)
    pred = dtree.predict(Xt)
    print "\n Complete vector size: ", len(X[0])
    print(classification_report(yt,pred))
    
    #remove irrelevant features
    quan = np.quantile(importances, rate)
    indices = np.where(importances < quan)
    X = np.delete(X, indices[0], axis=1)
    Xt = np.delete(Xt, indices[0], axis=1)

    dtree.fit(X,y)
    pred = dtree.predict(Xt)
    print "Reduced vector size (less relevant feat. removed): ", len(X[0])
    print(classification_report(yt,pred))
