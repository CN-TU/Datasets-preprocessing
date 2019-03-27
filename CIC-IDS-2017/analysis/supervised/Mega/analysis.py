# Analysis of the UNSW-NB15 dataset with all learners after CAIA vector extraction
# FM & FIV, Oct 2018
print('---------------------------------------------------------------------------------------')
print('Analysis of the IDS2017 dataset with all learners after CAIA vector extraction')
print('FM & FIV, Oct 2018, http://cn.tuwien.ac.at')
print('---------------------------------------------------------------------------------------')


import pandas as pd
import numpy as np

# read datasets
print('Loading training and test datasets...')
training = pd.read_csv("mega_training.csv").fillna(0)
testing = pd.read_csv("mega_testing.csv").fillna(0)
feat_set = "MEGA"

#preparing results file
file = open("results-all.csv","w")
file.write(','.join(list(training)))
file.write("\n")
 
# preparing the data and label for test and training
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
dtree = DecisionTreeClassifier(min_samples_leaf=1,max_depth=15, random_state=0)
dtree.fit(X_train,y_train)

print('\nFeature importance')
feat_imp = np.multiply(dtree.feature_importances_,100)
print(feat_imp)
feat_imp.tofile(file, sep=",", format="%.3f")
file.write("\n")

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
dtree = DecisionTreeClassifier(min_samples_leaf=1,max_depth=15, random_state=0)
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

# importing metrics 
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


# DT training
print('\n####################### DT classification #######################')
dtree = DecisionTreeClassifier(min_samples_leaf=1,max_depth=15, random_state=0)
dtree.fit(X_train_pca,y_train)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
sc_tr = cross_validate(dtree, X_train_pca, y_train, scoring=scoring, cv=5, return_train_score=False)
sc_ts = cross_validate(dtree, X_test_pca, y_test, scoring=scoring, cv=5, return_train_score=False)
pred_train = dtree.predict(X_train_pca)
pred = dtree.predict(X_test_pca)

print('-------------TRAINING--------------------')
print('confusion matrix')
cm = confusion_matrix(y_train,pred_train)
print(cm)
print(classification_report(y_train,pred_train))
print('roc_auc:')
rauc = roc_auc_score(y_train, pred_train)
print(rauc)
file.write("feature set, algorithm, train/test, accuracy, precision, recall, f1score, roc_auc\n")
acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
file.write("%s, DT (training) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

print('\n-------------TEST--------------------')
print('confusion matrix')
cm = confusion_matrix(y_test,pred)
print(cm)
print(classification_report(y_test,pred))
print('roc_auc:')
rauc = roc_auc_score(y_test, pred)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
file.write("%s, DT (test) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

# Naive bayes
print('\n####################### NB classification #######################')
from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB(alpha=215.43086172344687)
gnb.fit(X_train_pca,y_train)

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
sc_tr = cross_validate(gnb, X_train_pca, y_train, scoring=scoring, cv=5, return_train_score=False)
sc_ts = cross_validate(gnb, X_test_pca, y_test, scoring=scoring, cv=5, return_train_score=False)
pred = gnb.predict(X_test_pca)
pred_train = gnb.predict(X_train_pca)

print('-------------TRAINING--------------------')
print('confusion matrix')
cm = confusion_matrix(y_train,pred_train)
print(cm)
print(classification_report(y_train,pred_train))
print('roc_auc:')
rauc = roc_auc_score(y_train, pred_train)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
file.write("%s, NB (training) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

print('\n-------------TEST--------------------')
print('confusion matrix')
cm = confusion_matrix(y_test,predict)
print(cm)
print(classification_report(y_test,predict))
print('roc_auc:')
rauc = roc_auc_score(y_test, predict)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
file.write("%s, NB (test) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

# l2 logistic regression
print('\n####################### L2 classification #######################')
from sklearn import linear_model

l_reg = linear_model.LogisticRegression(penalty='l2', C=47, tol=0.0013673469387755102, random_state=0)
l_reg.fit(X_train_pca,y_train)
sc_tr = cross_validate(l_reg, X_train_pca, y_train, scoring=scoring, cv=5, return_train_score=False)
sc_ts = cross_validate(l_reg, X_test_pca, y_test, scoring=scoring, cv=5, return_train_score=False)
pred = l_reg.predict(X_test_pca)
pred_train = l_reg.predict(X_train_pca)

print('-------------TRAINING--------------------')
print('confusion matrix')
cm = confusion_matrix(y_train,pred_train)
print(cm)
print(classification_report(y_train,pred_train))
print('roc_auc:')
rauc = roc_auc_score(y_train, pred_train)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
file.write("%s, L2 (training) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

print('\n-------------TEST--------------------')
print('confusion matrix')
cm = confusion_matrix(y_test,predict)
print(cm)
print(classification_report(y_test,predict))
print('roc_auc:')
rauc = roc_auc_score(y_test, predict)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
file.write("%s, L2 (test) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

# RF
print('\n####################### RF classification #######################')
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=int((1+num_features/2)), min_samples_leaf=1,max_depth=15, random_state=0)
rfc.fit(X_train_pca,y_train)
sc_tr = cross_validate(rfc, X_train_pca, y_train, scoring=scoring, cv=5, return_train_score=False)
sc_ts = cross_validate(rfc, X_test_pca, y_test, scoring=scoring, cv=5, return_train_score=False)
pred = rfc.predict(X_test_pca)
pred_train = rfc.predict(X_train_pca)

print('-------------TRAINING--------------------')
print('confusion matrix')
cm = confusion_matrix(y_train,pred_train)
print(cm)
print(classification_report(y_train,pred_train))
print('roc_auc:')
rauc = roc_auc_score(y_train, pred_train)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
file.write("%s, RF (training) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

print('\n-------------TEST--------------------')
print('confusion matrix')
cm = confusion_matrix(y_test,pred)
print(cm)
print(classification_report(y_test,pred))
print('roc_auc:')
rauc = roc_auc_score(y_test, pred)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
file.write("%s, RF (test) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

# neural networks
print('\n####################### NN classification #######################')
from sklearn.neural_network import MLPClassifier

#{'beta_1': 0.58571428571428563, 'beta_2': 0.68367346938775508, 'hidden_layer_sizes': 18, 'epsilon': 2.3299518105153718e-09, 'alpha': 0.0064285714285714285, 'learning_rate_init': 0.038163265306122456}
#{'beta_1': 0.4281, 'beta_2': 0.4995, 'hidden_layer_sizes': 18, 'epsilon': 3.728e-08, 'alpha': 0.0107, 'learning_rate_init': 0.0139, hidden_layer_sizes=(num_features+1,2*num_features+1,1)}
# {'beta_1': 0.45510204081632649, 'beta_2': 0.65102040816326523, 'hidden_layer_sizes': array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]), 'epsilon': 3.23745754281764e-07, 'alpha': 0.0070408163265306117, 'learning_rate_init': 0.013673469387755103}
mlp = MLPClassifier(hidden_layer_sizes=(num_features,87,1),early_stopping=True,
	alpha=0.005,
	learning_rate_init=0.017346938775510204,
	epsilon=1.7575106248547893e-09,
	beta_1=0.6836734693877551,
	beta_2=0.6591836734693877, random_state=0)
mlp.fit(X_train_pca,y_train)
sc_tr = cross_validate(mlp, X_train_pca, y_train, scoring=scoring, cv=5, return_train_score=False)
sc_ts = cross_validate(mlp, X_test_pca, y_test, scoring=scoring, cv=5, return_train_score=False)
pred = mlp.predict(X_test_pca)
pred_train = mlp.predict(X_train_pca)

print('-------------TRAINING--------------------')
print('confusion matrix')
cm = confusion_matrix(y_train,pred_train)
print(cm)
print(classification_report(y_train,pred_train))
print('roc_auc:')
rauc = roc_auc_score(y_train, pred_train)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
file.write("%s, NN (training) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))

print('\n-------------TEST--------------------')
print('confusion matrix')
cm = confusion_matrix(y_test,predict)
print(cm)
print(classification_report(y_test,predict))
print('roc_auc:')
rauc = roc_auc_score(y_test, predict)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
file.write("%s, NN (test) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))
# SVM
print('\n####################### SVM classification #######################')
from sklearn.svm import SVC

svc = SVC(C=1.3636363636363638, gamma=3070.7070707070707, max_iter=400, random_state=0)
# {'C': 3.4545454545454546, 'gamma': 3474.7474747474748}
svc.fit(X_train_pca,y_train)
sc_tr = cross_validate(svc, X_train_pca, y_train, scoring=scoring, cv=5, return_train_score=False)
sc_ts = cross_validate(svc, X_test_pca, y_test, scoring=scoring, cv=5, return_train_score=False)
pred = svc.predict(X_test_pca)
pred_train = svc.predict(X_train_pca)

print('-------------TRAINING--------------------')
print('confusion matrix')
cm = confusion_matrix(y_train,pred_train)
print(cm)
print(classification_report(y_train,pred_train))
print('roc_auc:')
rauc = roc_auc_score(y_train, pred_train)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_tr['test_precision'].mean(), sc_tr['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_tr['test_recall'].mean(), sc_tr['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_tr['test_f1'].mean(), sc_tr['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_tr['test_roc_auc'].mean(), sc_tr['test_roc_auc'].std() * 2)
file.write("%s, SVM (training) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))


print('\n-------------TEST--------------------')
print('confusion matrix')
cm = confusion_matrix(y_test,predict)
print(cm)
print(classification_report(y_test,predict))
print('roc_auc:')
rauc = roc_auc_score(y_test, predict)
print(rauc)
acc = "%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2)
prec = "%0.3f (+/- %0.3f)" % (sc_ts['test_precision'].mean(), sc_ts['test_precision'].std() * 2)
rec = "%0.3f (+/- %0.3f)" % (sc_ts['test_recall'].mean(), sc_ts['test_recall'].std() * 2)
f1 = "%0.3f (+/- %0.3f)" % (sc_ts['test_f1'].mean(), sc_ts['test_f1'].std() * 2)
rauc = "%0.3f (+/- %0.3f)" % (sc_ts['test_roc_auc'].mean(), sc_ts['test_roc_auc'].std() * 2)
file.write("%s, SVM (test) - Acc: %s, Prec: %s, Rec: %s, F1: %s, Roc-Auc:%s\n" % (feat_set, acc, prec, rec, f1, rauc))


file.close()
