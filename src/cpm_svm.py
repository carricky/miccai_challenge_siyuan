#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:41:35 2019

@author: Wei Dai 
"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, SelectFdr
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc
from scipy import stats

import numpy as np
import scipy.io as sio
import h5py
import pandas as pd

import time
import matplotlib.pyplot as plt 

# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Prepare the training data
train_mats_aal_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/data/all_mats_aal.mat"
train_mats_cc_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/data/all_mats_cc.mat"
train_mats_ho_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/data/all_mats_ho.mat"
train_label_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/data/all_label.mat"

train_mats_aal_temp = sio.loadmat(train_mats_aal_dir) # this deals with -v7.3 .mat file
train_mats_aal = train_mats_aal_temp["all_mats_aal"]
train_mats_aal = np.array(train_mats_aal)

train_mats_cc_temp = sio.loadmat(train_mats_cc_dir) # this deals with -v7.3 .mat file
train_mats_cc = train_mats_cc_temp["all_mats_cc"]
train_mats_cc = np.array(train_mats_cc)

train_mats_ho_temp = sio.loadmat(train_mats_ho_dir) # this deals with -v7.3 .mat file
train_mats_ho = train_mats_ho_temp["all_mats_ho"]
train_mats_ho = np.array(train_mats_ho)

train_label = sio.loadmat(train_label_dir)
train_label = train_label["all_label"]
train_label = np.reshape(train_label, (-1,))

# Calculate edges for all parcellations for training data
num_sub = np.shape(train_mats_aal)[2]
num_node_aal = np.shape(train_mats_aal)[0]
num_edge_aal = num_node_aal * (num_node_aal - 1) // 2
num_node_ho = np.shape(train_mats_ho)[0]
num_edge_ho = num_node_ho * (num_node_ho - 1) // 2
num_node_cc = np.shape(train_mats_cc)[0]
num_edge_cc = num_node_cc * (num_node_cc - 1) // 2

train_edges = np.zeros([num_edge_aal + num_edge_ho + num_edge_cc, num_sub])

for i_sub in range(num_sub):
    iu_aal = np.triu_indices(num_node_aal, 1)
    iu_ho = np.triu_indices(num_node_ho, 1)
    iu_cc = np.triu_indices(num_node_cc, 1)
    train_edges[0:num_edge_aal, i_sub] = train_mats_aal[iu_aal[0], iu_aal[1], i_sub]
    train_edges[num_edge_aal:(num_edge_aal+num_edge_ho), i_sub] = train_mats_ho[iu_ho[0], iu_ho[1], i_sub]
    train_edges[(num_edge_aal+num_edge_ho):(num_edge_aal+num_edge_ho+num_edge_cc), i_sub] = train_mats_cc[iu_cc[0], iu_cc[1], i_sub]

train_edges_aal = train_edges[0:num_edge_aal, ]
train_edges_ho = train_edges[num_edge_aal:(num_edge_aal+num_edge_ho),]
train_edges_cc = train_edges[(num_edge_aal+num_edge_ho):(num_edge_aal+num_edge_ho+num_edge_cc), ]


# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Prepare the validation data
validation_mats_aal_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_aal.mat"
validation_mats_cc_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_cc.mat"
validation_mats_ho_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_ho.mat"
validation_label_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_label.mat"

validation_mats_aal_temp = sio.loadmat(validation_mats_aal_dir) # this deals with -v7.3 .mat file
validation_mats_aal = validation_mats_aal_temp["all_mats_aal"]
validation_mats_aal = np.array(validation_mats_aal)

validation_mats_cc_temp = sio.loadmat(validation_mats_cc_dir) # this deals with -v7.3 .mat file
validation_mats_cc = validation_mats_cc_temp["all_mats_cc"]
validation_mats_cc = np.array(validation_mats_cc)

validation_mats_ho_temp = sio.loadmat(validation_mats_ho_dir) # this deals with -v7.3 .mat file
validation_mats_ho = validation_mats_ho_temp["all_mats_ho"]
validation_mats_ho = np.array(validation_mats_ho)

validation_label = sio.loadmat(validation_label_dir)
validation_label = validation_label["all_label"]
validation_label = np.reshape(validation_label, (-1,))

# Calculate edges for all parcellations for validationing data
num_sub_validation = np.shape(validation_mats_aal)[2]

validation_edges = np.zeros([num_edge_aal + num_edge_ho + num_edge_cc, num_sub_validation])

for i_sub in range(num_sub_validation):
    iu_aal = np.triu_indices(num_node_aal, 1)
    iu_ho = np.triu_indices(num_node_ho, 1)
    iu_cc = np.triu_indices(num_node_cc, 1)
    validation_edges[0:num_edge_aal, i_sub] = validation_mats_aal[iu_aal[0], iu_aal[1], i_sub]
    validation_edges[num_edge_aal:(num_edge_aal+num_edge_ho), i_sub] = validation_mats_ho[iu_ho[0], iu_ho[1], i_sub]
    validation_edges[(num_edge_aal+num_edge_ho):(num_edge_aal+num_edge_ho+num_edge_cc), i_sub] = validation_mats_cc[iu_cc[0], iu_cc[1], i_sub]

validation_edges_aal = validation_edges[0:num_edge_aal, ]
validation_edges_ho = validation_edges[num_edge_aal:(num_edge_aal+num_edge_ho),]
validation_edges_cc = validation_edges[(num_edge_aal+num_edge_ho):(num_edge_aal+num_edge_ho+num_edge_cc), ]


# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Construct Model on the Training Dataset

## PCA on edges across all parcellations and then use SVM

pipe_steps = [('scalar', StandardScaler()), ('pca', PCA()), ("clf", SVC(kernel = "rbf", probability=True))]

check_parameters = {
        'pca__n_components': [2, 4, 10, 50, 100],
        'clf__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
        }
pipeline = Pipeline(pipe_steps)
cpm_svm_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
cpm_svm_grid.fit(train_edges.T, train_label)

## Best model is 2 components with gamma = 0.001, and prediction accuracy is 0.555


# ----------------------------------------------------------------------------------------- #
## Validate the best model on the validation dataset
## ## Evaluation Measures: Calculate accuracy, auc, sensitivity, specificity, recall, F-measure
y_predict = cpm_svm_grid.predict(validation_edges.T)
accuracy = accuracy_score(validation_label,y_predict) ## Prediction accuracy
prob_predict = cpm_svm_grid.predict_proba(validation_edges.T)
prob_predict = prob_predict.max(axis = 1)
fpr, tpr, thresholds = metrics.roc_curve(np.array(validation_label), np.array(prob_predict))
metrics.auc(fpr, tpr) ## AUC values
cmatrix = metrics.confusion_matrix(validation_label, y_predict) ## confusion matrix
sens = cmatrix[1, 1] / (cmatrix[1, 0] + cmatrix[1, 1])
spec = cmatrix[0, 0] / (cmatrix[0, 0] + cmatrix[0, 1])

print(metrics.classification_report(validation_label, y_predict))



# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Majority Vote
pipe_steps = [('scalar', StandardScaler()), ('pca', PCA()), ("clf", SVC(kernel = "rbf", probability=True))]

check_parameters = {
        'pca__n_components': [2, 10, 50],
        'clf__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
        }
pipeline = Pipeline(pipe_steps)
aal_svm_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
ho_svm_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
cc_svm_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
aal_svm_grid.fit(train_edges_aal.T, train_label)
ho_svm_grid.fit(train_edges_ho.T, train_label)
cc_svm_grid.fit(train_edges_cc.T, train_label)


aal_predict = aal_svm_grid.predict(validation_edges_aal.T)
aal_predict_prob = aal_svm_grid.predict_proba(validation_edges_aal.T)
ho_predict = ho_svm_grid.predict(validation_edges_ho.T)
ho_predict_prob = ho_svm_grid.predict_proba(validation_edges_ho.T)
cc_predict = cc_svm_grid.predict(validation_edges_cc.T)
cc_predict_prob = cc_svm_grid.predict_proba(validation_edges_cc.T)

majority_predict = np.concatenate((aal_predict, ho_predict, cc_predict))
majority_predict = np.reshape(majority_predict, [40, 3])
majority_vote = stats.mode(majority_predict, axis = 1)
majority_vote = majority_vote.mode

majority_prob = np.concatenate((aal_predict_prob.max(axis = 1), ho_predict_prob.max(axis = 1), cc_predict_prob.max(axis = 1)))
majority_prob = np.reshape(majority_prob, [40, 3])
majority_prob = stats.mode(majority_prob, axis = 1)
majority_prob = majority_prob.mode

# ----------------------------------------------------------------------------------------- #
## Evaluation Measures
accuracy_major = accuracy_score(validation_label,majority_vote)
print(metrics.classification_report(validation_label, majority_vote))

fpr, tpr, thresholds = metrics.roc_curve(np.array(validation_label), np.array(majority_prob))
metrics.auc(fpr, tpr) ## AUC values for majority vote method
cmatrix_major = metrics.confusion_matrix(validation_label, majority_vote) ## confusion matrix
sens_major = cmatrix_major[1, 1] / (cmatrix_major[1, 0] + cmatrix_major[1, 1])
spec_major = cmatrix_major[0, 0] / (cmatrix_major[0, 0] + cmatrix_major[0, 1])


# ----------------------------------------------------------------------------------------- #
# -------------------------------------Failed ?------------------------------ #
## Univariate feature selection on edges across all parcellations and then use SVM
pipe_steps = [('selection', SelectFdr(score_func = mutual_info_classif)), ('clf', SVC(kernel = "rbf"))]

check_parameters = {
        'selection__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'clf__gamma': [0.01, 0.05, 0.1, 0.5, 1, 5]
        }
pipeline = Pipeline(pipe_steps)
cpm_svm_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
cpm_svm_grid.fit(train_edges.T, train_label)


# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Visulize PCA
scaler1 = StandardScaler()
scaler1.fit(train_edges.T)
feature_scaled = scaler1.transform(train_edges.T)

pca1 = PCA(n_components = 4)
pca1.fit(feature_scaled)
feature_scaled_pca = pca1.transform(feature_scaled) 

ex_variance=np.var(feature_scaled_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print (ex_variance_ratio) # 1st PC explains 80% variance 

## 1st and 2nd PCs plot
Xax=feature_scaled_pca[:,0]
Yax=feature_scaled_pca[:,1]
labels = train_label
cdict={0:'red',1:'green'}
labl={0:'Control',1:'ADHD'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l],marker=marker[l],alpha=alpha[l])
# for loop ends
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()

## 3rd and 4th PCs plot
Xax=feature_scaled_pca[:,2]
Yax=feature_scaled_pca[:,3]
labels = train_label
cdict={0:'red',1:'green'}
labl={0:'Control',1:'ADHD'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l],marker=marker[l],alpha=alpha[l])
# for loop ends
plt.xlabel("Third Principal Component",fontsize=14)
plt.ylabel("Fourth Principal Component",fontsize=14)
plt.legend()
plt.show()