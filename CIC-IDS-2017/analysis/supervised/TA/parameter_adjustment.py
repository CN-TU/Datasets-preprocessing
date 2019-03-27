
print('---------------------------------------------------------------------------------------')
print('Parameter adjustment for learners, IDS-2017 dataset after TA vector extraction')
print('FM & FIV, Oct 2018, http://cn.tuwien.ac.at')
print('---------------------------------------------------------------------------------------')

import pandas as pd
import numpy as np

# read datasets
testing = pd.read_csv("TA_testing.csv").fillna(0)
training = pd.read_csv("TA_training.csv").fillna(0)

#preparing results file
X_train = training.drop('Label', axis = 1)
X_test = testing.drop('Label', axis = 1)
y_train = training['Label']
y_test = testing['Label']

del training
del testing

print('Training and test dataset loaded')

# minmax scalling
from sklearn.preprocessing import MinMaxScaler
scalermm = MinMaxScaler()
scalermm.fit(X_train)
X_train = scalermm.transform(X_train)
X_test = scalermm.transform(X_test)

# DT to extract feature importance
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=2018)
dtree.fit(X_train,y_train)

print('\nFeature importance')
feat_imp = np.multiply(dtree.feature_importances_,100)
print(feat_imp)

# feature selection, remove irrelevant features
indices = np.where(dtree.feature_importances_ == 0)
X_train = np.delete(X_train, indices[0], axis=1)
X_test = np.delete(X_test, indices[0], axis=1)

print('\nIrrelevant features deleted')
print(indices[0])

# center the data (- mean)
from sklearn.preprocessing import StandardScaler
scalerm = StandardScaler(with_std = False)
scalerm.fit(X_train)
X_train = scalerm.transform(X_train)
X_test = scalerm.transform(X_test)

# apply PCA
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print('\nSpace transformation based on PCA')
print(np.multiply(pca.explained_variance_ratio_,100))

# Check feature importance after PCA
from sklearn.model_selection import cross_validate
dtree = DecisionTreeClassifier(random_state=2018)
dtree.fit(X_train_pca,y_train)
predict_train = dtree.predict(X_train_pca)
predict = dtree.predict(X_test_pca)
print('\nImportance of features after PCA')
print(np.multiply(dtree.feature_importances_,100))

# feature selection, remove irrelevant features after space transformations
indices = np.where(dtree.feature_importances_ == 0)
X_train_pca = np.delete(X_train_pca, indices[0], axis=1)
X_test_pca = np.delete(X_test_pca, indices[0], axis=1)
num_features=len(X_train[0])
print('\nIrrelevant features deleted')
print(indices[0])

## Stratified sampling for reducing parameter search times 
print('\nTaking subset for parameter search (10%) ')
from sklearn.model_selection import train_test_split
X_train_little, X_test_little, y_train_little, y_test_little = train_test_split(X_train_pca, y_train, test_size=0.90, random_state=2018, stratify=y_train)

from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from evolutionary_search import EvolutionaryAlgorithmSearchCV

# Selecting learner to adjust
print('\nSelect learner to adjust: ')
print('1 - SVM')
print('2 - MLP')
print('3 - logistic regression')
print('4 - NB')
print('5 - RF')
#learner = input("Option? ")
learner = True

if learner:
	# SVM
	from sklearn.svm import SVC
	C_range = np.linspace(1, 10, 100)
	gamma_range = np.linspace(3000, 4000, 100)
	param_dist = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
	rnds = EvolutionaryAlgorithmSearchCV(estimator=SVC(max_iter=200),
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
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	print(rnds.best_score_)
	print('\nC: ')
	print(rnds.best_estimator_.C)
	print('\ngamma: ')
	print(rnds.best_estimator_.gamma)
	# apply best parameters
	svc = SVC(max_iter=400, C=rnds.best_estimator_.C, gamma=rnds.best_estimator_.gamma)
	svc.fit(X_train_pca,y_train)
	pred = svc.predict(X_test_pca)
	pred_train = svc.predict(X_train_pca)
if learner:
	# NNs
	from sklearn.neural_network import MLPClassifier
	# prepare parameter grid
	alpha_range = np.linspace(0.005, 0.015, 50)
	learning_rate_range = np.linspace(0.01, 0.07, 50)
	epsilon_range = np.logspace(-9, -6, 50)
	beta_1_range = np.linspace(0.3, 0.7, 50)
	beta_2_range = np.linspace(0.3, 0.7, 50)
	a=int((num_features+1)/2)
	b=int((num_features+1)/2+10)
	med_layer_range = np.arange(a,b)
	param_dist = dict(alpha=alpha_range, 
		hidden_layer_sizes=(num_features, med_layer_range, 1),
		learning_rate_init=learning_rate_range, 
		epsilon=epsilon_range,
		beta_1=beta_1_range,
		beta_2=beta_2_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
	rnds = EvolutionaryAlgorithmSearchCV(
			estimator=MLPClassifier(early_stopping=True),
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
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	print(rnds.best_score_)
	print('\nalpha: ')
	print(rnds.best_estimator_.alpha)
	print('\nhidden_layer_sizes: ')
	print(rnds.best_estimator_.hidden_layer_sizes)
	print('learning_rate: ')
	print(rnds.best_estimator_.learning_rate_init)
	print('epsilon: ')
	print(rnds.best_estimator_.epsilon)
	print('beta1: ')
	print(rnds.best_estimator_.beta_1)
	print('beta2: ')
	print(rnds.best_estimator_.beta_2)
	# apply best parameters
	mlp = MLPClassifier(hidden_layer_sizes=rnds.best_estimator_.hidden_layer_sizes,
		early_stopping=True,
		alpha=rnds.best_estimator_.alpha, 
		learning_rate_init=rnds.best_estimator_.learning_rate_init,
		epsilon=rnds.best_estimator_.epsilon,
		beta_1=rnds.best_estimator_.beta_1,
		beta_2=rnds.best_estimator_.beta_2)
	mlp.fit(X_train_pca,y_train)
	pred = mlp.predict(X_test_pca)
	pred_train = mlp.predict(X_train_pca)
if learner:
	# Logistic regr.
	from sklearn import linear_model
	# prepare parameter grid
	C_range = np.linspace(1, 50, 50)
	tol_range = np.linspace(0.001, 0.01, 50)
	param_dist = dict(tol=tol_range, C=C_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
	rnds = EvolutionaryAlgorithmSearchCV(
			estimator=linear_model.LogisticRegression(penalty='l2'),
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
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	print(rnds.best_score_)
	print('\nC: ')
	print(rnds.best_estimator_.C)
	print('tol: ')
	print(rnds.best_estimator_.tol)
	# apply best parameters
	l2r = linear_model.LogisticRegression(C=rnds.best_estimator_.C, tol=rnds.best_estimator_.tol)
	l2r.fit(X_train_pca,y_train)
	pred = l2r.predict(X_test_pca)
	pred_train = l2r.predict(X_train_pca)
if learner:
	# Naive Bayes.
	from sklearn.naive_bayes import BernoulliNB
	# prepare parameter grid
	alpha_range = np.linspace(0, 500, 500)
	param_dist = dict(alpha=alpha_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
	rnds = EvolutionaryAlgorithmSearchCV(
			estimator=BernoulliNB(),
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
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	print(rnds.best_score_)
	print('\nalpha: ')
	print(rnds.best_estimator_.alpha)
	# apply best parameters
	nbb = BernoulliNB(alpha=rnds.best_estimator_.alpha)
	nbb.fit(X_train_pca,y_train)
	pred = nbb.predict(X_test_pca)
	pred_train = nbb.predict(X_train_pca)
if learner:
	# Random Forest
	from sklearn.ensemble import RandomForestClassifier
	min_samples_leaf_r = np.round(np.linspace(1, 80, 30))
	min_samples_leaf_range = min_samples_leaf_r.astype(int)
	max_depth_range = np.round(np.linspace(5, 15, 30))
	param_dist = dict(min_samples_leaf=min_samples_leaf_range, max_depth=max_depth_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
	rnds = EvolutionaryAlgorithmSearchCV(
			estimator=RandomForestClassifier(n_estimators=(1+num_features/2)),
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
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	print(rnds.best_score_)
	print('min_samples_leaf: ')
	print(rnds.best_estimator_.min_samples_leaf)
	print('max_depth: ')
	print(rnds.best_estimator_.max_depth)
	# apply best parameters
	rf = RandomForestClassifier(n_estimators=(1+num_features/2), 
		min_samples_leaf = rnds.best_estimator_.min_samples_leaf, 
		max_depth = rnds.best_estimator_.max_depth)
	rf.fit(X_train_pca,y_train)
	pred = rf.predict(X_test_pca)
	pred_train = rf.predict(X_train_pca)
#else:
	#print('\nInvalid option!!')
	#quit()

from sklearn.metrics import classification_report,confusion_matrix
print('\n####################### best parameters classif. #######################')
print('-------------TRAINING--------------------')
print('confusion matrix')
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print('-------------TEST--------------------')
print('confusion matrix')
print(confusion_matrix(y_train,pred_train))
print(classification_report(y_train,pred_train))

