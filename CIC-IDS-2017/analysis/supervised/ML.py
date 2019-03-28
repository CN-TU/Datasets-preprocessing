# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2019, Institute of Telecommunications, TU Wien
#
# Name        : ML.py
# Description : parameters tunning and ML analysis for datasets
# Author      : Fares Meghdouri
#
# Notes : additionnaly to this script, the training data (under x_training.csv) 
#         and testing data (under testing.csv) need to be included in the same
#         directory. the variable VECTOR_NAME also need to be changed then to "x"
#
#
#******************************************************************************

# generics
import pandas as pd
import numpy as np
import pickle

# genetic search
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from evolutionary_search import EvolutionaryAlgorithmSearchCV

#


#******************************************************************************

# ignore sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#******************************************************************************

VECTOR_NAME  = "mega" # CHANGE THIS TO YOUR FEATURE VECTOR
DATASET_NAME = ""
SEED         = 2019
SCORING 	 = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
TUNE_ONLY 	 = False
features     = []

#******************************************************************************

print('---------------------------------------------------------------------------------------')
print('Analysis of the {} dataset with all learners after {} vector extraction'.format(DATASET_NAME, VECTOR_NAME))
print('FM, Mar 2019, http://cn.tuwien.ac.at')
print('---------------------------------------------------------------------------------------')

def read_data():
	# read datasets
	training = pd.read_csv("{}_training.csv".format(VECTOR_NAME)).fillna(0)
	testing  = pd.read_csv("{}_testing.csv".format(VECTOR_NAME)).fillna(0)

	# preparing the data and label for test and training
	X_train = training.drop('Label', axis = 1)
	X_test  = testing.drop('Label', axis = 1)
	y_train = training['Label']
	y_test  = testing['Label']

	features = ",".join(list(X_train))

	del training
	del testing

	return X_train, X_test, y_train, y_test, features


def min_max_scaling(X_train, X_test):
	# minmax scalling
	from sklearn.preprocessing import MinMaxScaler
	scalermm = MinMaxScaler()
	scalermm.fit(X_train)
	X_train  = scalermm.transform(X_train)
	X_test   = scalermm.transform(X_test)

	return X_train, X_test


def standard_scaling(X_train, X_test):
	from sklearn.preprocessing import StandardScaler
	scalerm = StandardScaler(with_std = False)
	scalerm.fit(X_train)
	X_train = scalerm.transform(X_train)
	X_test  = scalerm.transform(X_test)

	return X_train, X_test

def feature_importance(pca, X_train, X_test, y_train, min_samples_leaf, max_depth, _with=False,):
	# DT to extract feature importance
	from sklearn.tree import DecisionTreeClassifier
	if _with:
		file  = open("feature_importance_with_tuning_{}.csv".format(pca),"w")
		dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, random_state=SEED)
	else:
		file  = open("feature_importance_without_tuning_{}.csv".format(pca),"w")
		dtree = DecisionTreeClassifier(random_state=SEED)
	dtree.fit(X_train,y_train)

	feat_imp = np.multiply(dtree.feature_importances_,100)
	file.write(features + "\n")
	file.write("\n")
	feat_imp.tofile(file, sep=",", format="%.3f")
	file.write("\n")
	file.close()

	# feature selection, remove irrelevant features
	indices = np.where(dtree.feature_importances_ == 0)
	X_train = np.delete(X_train, indices[0], axis=1)
	X_test  = np.delete(X_test, indices[0], axis=1)

	if _with:
		file = open("feature_selection_with_tuning_{}.csv".format(pca),"w")
	else:
		file = open("feature_selection_with_tuning_{}.csv".format(pca),"w")
	indices[0].tofile(file, sep=",", format="%.3f")
	file.write("\n")
	file.close()

	return X_train, X_test


def _pca(X_train, X_test):
	# apply PCA
	from sklearn.decomposition import PCA
	pca = PCA()
	pca.fit(X_train)
	X_train_pca = pca.transform(X_train)
	X_test_pca  = pca.transform(X_test)
	
	file        = open("pca_results.csv","w")
	np.multiply(pca.explained_variance_ratio_,100).tofile(file, sep=",", format="%.3f")
	file.write("\n")
	file.close()

	return X_train_pca, X_test_pca


def reducing(X_train_pca, y_train, size):
	# reduce the training set to reduce time
	from sklearn.model_selection import train_test_split
	X_train_little, X_test_little, y_train_little, y_test_little = train_test_split(X_train_pca, y_train, test_size=size, random_state=SEED, stratify=y_train)

	return X_train_little, y_train_little


def output_report(algo, y_train, pred_train, y_test, pred, sc_tr, sc_ts):
	from sklearn.metrics import classification_report,confusion_matrix
	from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

	text_file = open("{}_classification_report.txt".format(algo), "w")

	text_file.write('-------------TRAINING--------------------\n')
	text_file.write('confusion matrix\n')
	cm   = confusion_matrix(y_train,pred_train)
	text_file.write("{}".format(cm))
	text_file.write('\nclassification report:\n')
	text_file.write(classification_report(y_train,pred_train))
	text_file.write('\nroc_auc:\n')
	rauc = roc_auc_score(y_train, pred_train)
	text_file.write("{}".format(rauc))
	text_file.write("\n")
	acc  = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
	rec  = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
	f1   = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
	text_file.write("%s, %s (training) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (VECTOR_NAME, algo, acc, prec, rec, f1, rauc))


	text_file.write('\n-------------TEST--------------------\n')
	text_file.write('confusion matrix\n')
	cm   = confusion_matrix(y_test,pred)
	text_file.write("{}".format(cm))
	text_file.write('\nclassification report:\n')
	text_file.write(classification_report(y_test,pred))
	text_file.write('\nroc_auc:\n')
	rauc = roc_auc_score(y_test, pred)
	text_file.write("{}".format(rauc))
	text_file.write("\n")
	acc  = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
	prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
	rec  = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
	f1   = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
	rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
	text_file.write("%s, %s (test) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (VECTOR_NAME, algo, acc, prec, rec, f1, rauc))


def RF_DT(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=False):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.model_selection import cross_validate
	min_samples_leaf_range = np.round(np.linspace(1, 20, 20)).astype(int)
	max_depth_range 	   = np.round(np.linspace(1, 15, 15)).astype(int)
	param_dist 			   = dict(min_samples_leaf=min_samples_leaf_range, max_depth=max_depth_range)
	num_features		   = len(X_train_little[0])
	cv 					   = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
	rnds 				   = EvolutionaryAlgorithmSearchCV( estimator     	    = RandomForestClassifier(n_estimators=int((1+num_features/2))),
															params              = param_dist,
															scoring             = "f1",
															cv                  = cv,
															verbose				= 1,
															population_size	    = 50,
															gene_mutation_prob  = 0.10,
															gene_crossover_prob = 0.5,
															tournament_size		= 3,
															generations_number	= 6,
															n_jobs				= 4)
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	file = open("RF_DT_best_parameters.txt","w")
	file.write("{}\n".format(rnds.best_score_))
	file.write('min_samples_leaf: {}\n'.format(rnds.best_estimator_.min_samples_leaf))
	file.write('max_depth: {}\n'.format(rnds.best_estimator_.max_depth))
	file.close()

	if not tune_only:
		# apply best parameters RF
		rfc = RandomForestClassifier(n_estimators    = int((1+num_features/2)), 
									min_samples_leaf = rnds.best_estimator_.min_samples_leaf, 
									max_depth        = rnds.best_estimator_.max_depth,
									random_state     = SEED)
		rfc.fit(X_train_pca,y_train)
		sc_tr      = cross_validate(rfc, X_train_pca, y_train, scoring=SCORING, cv=5, return_train_score=False)
		sc_ts      = cross_validate(rfc, X_test_pca, y_test, scoring=SCORING, cv=5, return_train_score=False)
		pred       = rfc.predict(X_test_pca)
		pred_train = rfc.predict(X_train_pca)

		output_report("RF", y_train, pred_train, y_test, pred, sc_tr, sc_ts)

		# apply best parameters DT
		dtc = DecisionTreeClassifier(min_samples_leaf = rnds.best_estimator_.min_samples_leaf, 
									 max_depth        = rnds.best_estimator_.max_depth,
									 random_state     = SEED)
		dtc.fit(X_train_pca,y_train)
		sc_tr      = cross_validate(dtc, X_train_pca, y_train, scoring=SCORING, cv=5, return_train_score=False)
		sc_ts      = cross_validate(dtc, X_test_pca, y_test, scoring=SCORING, cv=5, return_train_score=False)
		pred       = dtc.predict(X_test_pca)
		pred_train = dtc.predict(X_train_pca)

		output_report("DT", y_train, pred_train, y_test, pred, sc_tr, sc_ts)


def SVM(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=False):
	from sklearn.svm import SVC
	from sklearn.model_selection import cross_validate
	C_range     = np.linspace(1, 10, 101)
	gamma_range = np.linspace(3000, 4000, 100)
	param_dist  = dict(gamma=gamma_range, C=C_range)
	cv 			= StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
	rnds 	    = EvolutionaryAlgorithmSearchCV(estimator     	    = SVC(max_iter=200),
												params              = param_dist,
												scoring             = "f1",
												cv                  = cv,
												verbose				= 1,
												population_size	    = 50,
												gene_mutation_prob  = 0.10,
												gene_crossover_prob = 0.5,
												tournament_size		= 3,
												generations_number	= 6,
												n_jobs				= 4)
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	file = open("SVM_best_parameters.txt","w")
	file.write("{}\n".format(rnds.best_score_))
	file.write('C: {}\n'.format(rnds.best_estimator_.C))
	file.write('gamma: {}\n'.format(rnds.best_estimator_.gamma))
	file.close()

	if not tune_only:
		# apply best parameters
		svc = SVC(max_iter=200, C=rnds.best_estimator_.C, gamma=rnds.best_estimator_.gamma, random_state=SEED)
		svc.fit(X_train_pca,y_train)
		sc_tr      = cross_validate(svc, X_train_pca, y_train, scoring=SCORING, cv=5, return_train_score=False)
		sc_ts      = cross_validate(svc, X_test_pca, y_test, scoring=SCORING, cv=5, return_train_score=False)
		pred       = svc.predict(X_test_pca)
		pred_train = svc.predict(X_train_pca)

		output_report("SVM", y_train, pred_train, y_test, pred, sc_tr, sc_ts)

def NN(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=False):
	from sklearn.neural_network import MLPClassifier
	from sklearn.model_selection import cross_validate
	num_features = len(X_train_little[0])
	# prepare parameter grid
	alpha_range = np.linspace(0.005, 0.015, 50)
	learning_rate_range = np.linspace(0.01, 0.07, 50)
	epsilon_range = np.logspace(-9, -6, 50)
	beta_1_range = np.linspace(0.3, 0.7, 50)
	beta_2_range = np.linspace(0.3, 0.7, 50)
	a=int((num_features+1)/2)
	b=int((num_features+1)/2+10)
	med_layer_range = np.arange(a,b)
	param_dist = dict(alpha              = alpha_range, 
					  hidden_layer_sizes = (num_features, med_layer_range, 1),
					  learning_rate_init = learning_rate_range, 
					  epsilon            = epsilon_range,
					  beta_1             = beta_1_range,
					  beta_2             = beta_2_range)
	cv 	 = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
	rnds = EvolutionaryAlgorithmSearchCV(estimator     	    = MLPClassifier(early_stopping=True),
										params              = param_dist,
										scoring             = "f1",
										cv                  = cv,
										verbose				= 1,
										population_size	    = 50,
										gene_mutation_prob  = 0.10,
										gene_crossover_prob = 0.5,
										tournament_size		= 3,
										generations_number	= 6,
										n_jobs				= 4)
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	file = open("MLP_best_parameters.txt","w")
	file.write("{}\n".format(rnds.best_score_))
	file.write('alpha: {}\n'.format(rnds.best_estimator_.alpha))
	file.write('hidden_layer_sizes: {}\n'.format(rnds.best_estimator_.hidden_layer_sizes))
	file.write('learning_rate_init: {}\n'.format(rnds.best_estimator_.learning_rate_init))
	file.write('epsilon: {}\n'.format(rnds.best_estimator_.epsilon))
	file.write('beta_1: {}\n'.format(rnds.best_estimator_.beta_1))
	file.write('beta_2: {}\n'.format(rnds.best_estimator_.beta_2))
	file.close()

	if not tune_only:
		# apply best parameters
		mlp = MLPClassifier(hidden_layer_sizes=rnds.best_estimator_.hidden_layer_sizes,
							early_stopping=True,
							alpha=rnds.best_estimator_.alpha, 
							learning_rate_init=rnds.best_estimator_.learning_rate_init,
							epsilon=rnds.best_estimator_.epsilon,
							beta_1=rnds.best_estimator_.beta_1,
							beta_2=rnds.best_estimator_.beta_2,
							random_state=SEED)
		mlp.fit(X_train_pca,y_train)
		sc_tr      = cross_validate(mlp, X_train_pca, y_train, scoring=SCORING, cv=5, return_train_score=False)
		sc_ts      = cross_validate(mlp, X_test_pca, y_test, scoring=SCORING, cv=5, return_train_score=False)
		pred       = mlp.predict(X_test_pca)
		pred_train = mlp.predict(X_train_pca)

		output_report("MLP", y_train, pred_train, y_test, pred, sc_tr, sc_ts)


def NB(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=False):
	from sklearn.naive_bayes import BernoulliNB
	from sklearn.model_selection import cross_validate
	alpha_range = np.linspace(0, 500, 500)
	param_dist  = dict(alpha=alpha_range)
	cv 			= StratifiedShuffleSplit(n_splits=5, test_size=0.2)
	rnds 	    = EvolutionaryAlgorithmSearchCV(estimator     	    = BernoulliNB(),
												params              = param_dist,
												scoring             = "f1",
												cv                  = cv,
												verbose				= 1,
												population_size	    = 50,
												gene_mutation_prob  = 0.10,
												gene_crossover_prob = 0.5,
												tournament_size		= 3,
												generations_number	= 6,
												n_jobs				= 4)
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	file = open("NB_best_parameters.txt","w")
	file.write("{}\n".format(rnds.best_score_))
	file.write('alpha: {}\n'.format(rnds.best_estimator_.alpha))
	file.close()

	if not tune_only:
		# apply best parameters
		gnb = BernoulliNB(alpha=rnds.best_estimator_.alpha)
		gnb.fit(X_train_pca,y_train)
		sc_tr      = cross_validate(gnb, X_train_pca, y_train, scoring=SCORING, cv=5, return_train_score=False)
		sc_ts      = cross_validate(gnb, X_test_pca, y_test, scoring=SCORING, cv=5, return_train_score=False)
		pred       = gnb.predict(X_test_pca)
		pred_train = gnb.predict(X_train_pca)

		output_report("NB", y_train, pred_train, y_test, pred, sc_tr, sc_ts)

def LR2(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=False):
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import cross_validate
	C_range    = np.linspace(1, 50, 50)
	tol_range  = np.linspace(0.001, 0.01, 50)
	param_dist = dict(tol=tol_range, C=C_range)
	cv 			= StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
	rnds 	    = EvolutionaryAlgorithmSearchCV(estimator     	    = LogisticRegression(penalty='l2'),
												params              = param_dist,
												scoring             = "f1",
												cv                  = cv,
												verbose				= 1,
												population_size	    = 50,
												gene_mutation_prob  = 0.10,
												gene_crossover_prob = 0.5,
												tournament_size		= 3,
												generations_number	= 6,
												n_jobs				= 4)
	rnds.fit(X_train_little, y_train_little)
	# summarize the results of the random parameter search
	file = open("LR2_best_parameters.txt","w")
	file.write("{}\n".format(rnds.best_score_))
	file.write('C: {}\n'.format(rnds.best_estimator_.C))
	file.write('tol: {}\n'.format(rnds.best_estimator_.tol))
	file.close()

	if not tune_only:
		# apply best parameters
		l2r = LogisticRegression(C=rnds.best_estimator_.C, tol=rnds.best_estimator_.tol, random_state=SEED)
		l2r.fit(X_train_pca,y_train)
		sc_tr      = cross_validate(l2r, X_train_pca, y_train, scoring=SCORING, cv=5, return_train_score=False)
		sc_ts      = cross_validate(l2r, X_test_pca, y_test, scoring=SCORING, cv=5, return_train_score=False)
		pred       = l2r.predict(X_test_pca)
		pred_train = l2r.predict(X_train_pca)

		output_report("LR2", y_train, pred_train, y_test, pred, sc_tr, sc_ts)


def main():
	global features
	###### Preprocessing #######
	try:
		print('-->> Loading training and test datasets...')
		X_train, X_test, y_train, y_test = read_data()
		print('-->> Training and test dataset loaded.')
	except:
		print("Make sure to include your training and testing data under the same directory and, change the name in the script parameters")
		exit()

	X_train, X_test = min_max_scaling(X_train, X_test)
	print('-->> Data scaled.')
	
	X_train, X_test = feature_importance("", X_train, X_test, y_train, 0, 0, _with=False)
	print('-->> Feature selection done.')

	X_train, X_test = standard_scaling(X_train, X_test)
	print('-->> Standard scaling done.')

	X_train_pca, X_test_pca = _pca(X_train, X_test)
	print('-->> Space transformation based on PCA')

	X_train_pca, X_test_pca = feature_importance("after_pca", X_train_pca, X_test_pca, y_train , 0, 0, _with=False)
	print('-->> Feature selection after PCA done.')

	reducing_rate = 0.90
	X_train_little, y_train_little = reducing(X_train_pca, y_train, reducing_rate)
	print('-->> Taking a sub-portion of {} pecent of the data'.format((1-reducing_rate)*100))

	###### Tuning and Analysis #######
	print("*"*80)
	
	try:
		RF_DT(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=TUNE_ONLY)
	except Exception as e:
		print("-->> RF_DT issues: {}".format(e))

	print("*"*80)
	
	try:
		SVM(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=TUNE_ONLY)
	except Exception as e:
		print("-->> SVM issues: {}".format(e))

	print("*"*80)
	
	try:
		NN(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=TUNE_ONLY)
	except Exception as e:
		print("-->> NN issues: {}".format(e))

	print("*"*80)
	
	try:
		NB(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=TUNE_ONLY)
	except Exception as e:
		print("-->> NB issues: {}".format(e))

	print("*"*80)
	
	try:
		LR2(X_train_little, y_train_little, X_train_pca, X_test_pca, y_train, y_test, tune_only=TUNE_ONLY)
	except Exception as e:
		print("-->> LR2 issues: {}".format(e))

main()