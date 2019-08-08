#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:35:27 2019

@author: wei
"""

from __future__ import division
import glob
def findFiles(path): return sorted(glob.glob(path))
import numpy as np
import random
import math
import statistics
import sys
from numpy import dot
from numpy import inner
from numpy.linalg import norm
import math
import collections
import itertools
from sklearn.model_selection import KFold
import time
import os
import pandas as pd
import scipy.io as sio


from functools import reduce
from sklearn.impute import SimpleImputer
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pyprind
import sys
import pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn import tree
import functools
import numpy.ma as ma # for masked arrays
import pyprind
import random

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFdr, SelectPercentile
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


train_time_aal ={}
train_time_ho ={}
train_time_cc ={}
sub_name = []
# train_label = {}

eig_data_aal = {}
eig_data_ho = {}
eig_data_cc = {}

# Load Training Time Series Data and Calculate Eigenvectors and eigenvalues

for filename in findFiles('/Users/wei/Desktop/2019_CNI_TrainingRelease-master/Training/*'):
    for subfilename_aal in findFiles(filename+'/timeseries_aal*'):
        mats_aal = pd.read_csv(subfilename_aal, header = -1)
        label_aal = pd.read_csv(subfilename_aal.replace(subfilename_aal.split('/')[-1], 'phenotypic.csv'), header = 0)
        label_aal = label_aal['DX']
        
        regionnum_aal= len(mats_aal)
        mats_aal = np.array(mats_aal)
        
        sub = subfilename_aal.split('/')[-2]
        sub_name.append(sub)
        
        train_time_aal[sub] = (mats_aal, label_aal)
        
        
        # Calculate Eigenvectors and eigenvalues
        cor_aal = np.nan_to_num(np.corrcoef(mats_aal))
        eig_vals, eig_vecs = np.linalg.eig(cor_aal)
        
        # Make a list of (eigenvalue, eigenvector, norm_eigval) tuples
        sum_eigvals = np.sum(np.abs(eig_vals))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i], np.abs(eig_vals[i])/sum_eigvals)
                     for i in range(len(eig_vals))]
        
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        
        
        eig_data_aal[sub] = {'eigvals':np.array([ep[0] for ep in eig_pairs]),
                       'norm-eigvals':np.array([ep[2] for ep in eig_pairs]),
                       'eigvecs':[ep[1] for ep in eig_pairs]}
        
    for subfilename_ho in findFiles(filename+'/timeseries_ho*'):
        mats_ho = pd.read_csv(subfilename_ho, header = -1)
        label_ho = pd.read_csv(subfilename_ho.replace(subfilename_ho.split('/')[-1], 'phenotypic.csv'), header = 0)
        label_ho = label_ho['DX']
        regionnum_ho= len(mats_ho)
        mats_ho = np.array(mats_ho)
        
        sub = subfilename_ho.split('/')[-2]
        train_time_ho[sub] = (mats_ho, label_ho)
        
        
        # Calculate Eigenvectors and eigenvalues
        cor_ho = np.nan_to_num(np.corrcoef(mats_ho))
        eig_vals, eig_vecs = np.linalg.eig(cor_ho)
        
        # Make a list of (eigenvalue, eigenvector, norm_eigval) tuples
        sum_eigvals = np.sum(np.abs(eig_vals))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i], np.abs(eig_vals[i])/sum_eigvals)
                     for i in range(len(eig_vals))]
        
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        
        eig_data_ho[sub] = {'eigvals':np.array([ep[0] for ep in eig_pairs]),
                       'norm-eigvals':np.array([ep[2] for ep in eig_pairs]),
                       'eigvecs':[ep[1] for ep in eig_pairs]}
        
        
    for subfilename_cc in findFiles(filename+'/timeseries_cc*'):
        mats_cc = pd.read_csv(subfilename_cc, header = -1)
        label_cc = pd.read_csv(subfilename_cc.replace(subfilename_cc.split('/')[-1], 'phenotypic.csv'), header = 0)
        label_cc = label_cc['DX']
        regionnum_cc= len(mats_cc)
        mats_cc = np.array(mats_cc)
        
        sub = subfilename_cc.split('/')[-2]
        
        train_time_cc[sub] = (mats_cc, label_cc)
        
        
        # Calculate Eigenvectors and eigenvalues
        cor_cc = np.nan_to_num(np.corrcoef(mats_cc))
        eig_vals, eig_vecs = np.linalg.eig(cor_cc)
        
        # Make a list of (eigenvalue, eigenvector, norm_eigval) tuples
        sum_eigvals = np.sum(np.abs(eig_vals))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i], np.abs(eig_vals[i])/sum_eigvals)
                     for i in range(len(eig_vals))]
        
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        
        eig_data_cc[sub] = {'eigvals':np.array([ep[0] for ep in eig_pairs]),
                       'norm-eigvals':np.array([ep[2] for ep in eig_pairs]),
                       'eigvecs':[ep[1] for ep in eig_pairs]}

train_time_all = {}
eig_data_all = {}
for f in sub_name[11:]:
    all_time = np.concatenate((train_time_aal[f][0], train_time_ho[f][0], train_time_cc[f][0]))
    train_time_all[f] = (all_time, train_time_aal[f][1][0])
    
    cor_all = np.nan_to_num(np.corrcoef(all_time))
    eig_vals, eig_vecs = np.linalg.eig(cor_all)
        
    # Make a list of (eigenvalue, eigenvector, norm_eigval) tuples
    sum_eigvals = np.sum(np.abs(eig_vals))
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i], np.abs(eig_vals[i])/sum_eigvals)
                    for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    eig_data_all[f] = {'eigvals':np.array([ep[0] for ep in eig_pairs]),
                       'norm-eigvals':np.array([ep[2] for ep in eig_pairs]),
                       'eigvecs':[ep[1] for ep in eig_pairs]}
    
# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Load Connectome Training Data
train_mats_aal_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/data/all_mats_aal.mat"
train_mats_cc_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/data/all_mats_cc.mat"
train_mats_ho_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/data/all_mats_ho.mat"

train_mats_aal_temp = sio.loadmat(train_mats_aal_dir) # this deals with -v7.3 .mat file
train_mats_aal = train_mats_aal_temp["all_mats_aal"]
train_mats_aal = np.array(train_mats_aal)

train_mats_cc_temp = sio.loadmat(train_mats_cc_dir) # this deals with -v7.3 .mat file
train_mats_cc = train_mats_cc_temp["all_mats_cc"]
train_mats_cc = np.array(train_mats_cc)

train_mats_ho_temp = sio.loadmat(train_mats_ho_dir) # this deals with -v7.3 .mat file
train_mats_ho = train_mats_ho_temp["all_mats_ho"]
train_mats_ho = np.array(train_mats_ho)


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


train_corr_aal = {}
for sn in range(200):
    train_corr_aal[sub_name[sn]] = (train_edges_aal[:,sn], train_time_aal[sub_name[sn]][1][0])
    
train_corr_ho = {}
for sn in range(200):
    train_corr_ho[sub_name[sn]] = (train_edges_ho[:,sn], train_time_ho[sub_name[sn]][1][0])
    
train_corr_cc = {}
for sn in range(200):
    train_corr_cc[sub_name[sn]] = (train_edges_cc[:,sn], train_time_cc[sub_name[sn]][1][0])
    
train_corr_all = {}
for sn in range(11, 200):
    train_corr_all[sub_name[sn]] = (train_edges[:,sn], train_time_all[sub_name[sn]][1])


# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Prepare the validation data
validation_mats_aal_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_aal.mat"
validation_mats_cc_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_cc.mat"
validation_mats_ho_dir = "/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_ho.mat"
validation_label_dir = "/Users/wei/Desktop/2019_CNI_ValidationRelease-master/SupportingInfo/phenotypic_validation.csv"

validation_mats_aal_temp = sio.loadmat(validation_mats_aal_dir) # this deals with -v7.3 .mat file
validation_mats_aal = validation_mats_aal_temp["all_mats_aal"]
validation_mats_aal = np.array(validation_mats_aal)

validation_mats_cc_temp = sio.loadmat(validation_mats_cc_dir) # this deals with -v7.3 .mat file
validation_mats_cc = validation_mats_cc_temp["all_mats_cc"]
validation_mats_cc = np.array(validation_mats_cc)

validation_mats_ho_temp = sio.loadmat(validation_mats_ho_dir) # this deals with -v7.3 .mat file
validation_mats_ho = validation_mats_ho_temp["all_mats_ho"]
validation_mats_ho = np.array(validation_mats_ho)

validation_label = pd.read_csv(validation_label_dir, header = 0)
validation_label = validation_label["DX"]
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
# Data Augmentation Implement

# First Select half of the features
def get_regs(samplesnames,regnum, dataset):
    datas = []
    for sn in samplesnames:
        datas.append(dataset[sn][0])
    datas = np.array(datas)
    avg=[]
    for ie in range(datas.shape[1]):
        avg.append(np.mean(datas[:, ie]))
    avg=np.array(avg)
    highs=avg.argsort()[-regnum:][::-1]
    lows=avg.argsort()[:regnum][::-1]
    regions=np.concatenate((highs,lows),axis=0)
    return regions


# Function to calculate weight
def norm_weights(sub_names, eig_data):
    num_dim = len(eig_data[sub_name[11]]['eigvals'])
    norm_weights = np.zeros(shape=num_dim)
    for sub in sub_names:
        norm_weights += eig_data[sub]['norm-eigvals'] 
    return norm_weights


# Function to calculate Eros Similarity
def cal_similarity(d1, d2, weights, lim=None):
    res = 0.0
    if lim is None:
        weights_arr = weights.copy()
    else:
        weights_arr = weights[:lim].copy()
        weights_arr /= np.sum(weights_arr)
    for i,w in enumerate(weights_arr):
        res += w*np.inner(d1[i], d2[i])
    return res


class AugDataset(Dataset):
    def __init__(self, data=None, samples_list=None, 
                 augmentation=True, aug_factor=1, num_neighbs=5,
                 eig_data=None, similarity_fn=None, verbose=False,regs=None):
        self.regs=regs
        if data is not None:
            self.data = data.copy()
        if samples_list is None:
            self.flist = [f for f in self.data]
        else:
            self.flist = [f for f in samples_list]
            
        self.labels = np.array([self.data[f][1] for f in self.flist])
        current_flist = np.array(self.flist.copy())
        current_lab0_flist = current_flist[self.labels == 'Control']
        current_lab1_flist = current_flist[self.labels == 'ADHD']
        
        
        if augmentation:
            self.num_data = aug_factor * len(self.flist)
            self.neighbors = {}
            weights = norm_weights(self.flist, eig_data)
            for f in self.flist:
                label = self.data[f][1]
                candidates = (set(current_lab0_flist) if label == 'Control' else set(current_lab1_flist))
                candidates.remove(f)
                eig_f = eig_data[f]['eigvecs']
                sim_list = []
                for cand in candidates:
                    eig_cand = eig_data[cand]['eigvecs']
                    sim = similarity_fn(eig_f, eig_cand, weights)
                    sim_list.append((sim, cand))
                sim_list.sort(key=lambda x: x[0], reverse=True)
                self.neighbors[f] = [item[1] for item in sim_list[:num_neighbs]]#list(candidates)#[item[1] for item in sim_list[:num_neighbs]]
        
        else:
            self.num_data = len(self.flist)
            
    def __getitem__(self, index):
        if index < len(self.flist):
            fname = self.flist[index]
            data = self.data[fname][0].copy() #get_corr_data(fname, mode=cal_mode)    
            data = data[self.regs].copy()
            label = (self.labels[index])
            return (data, label)
        else:
            f1 = self.flist[index % len(self.flist)]
            d1, y1 = self.data[f1][0], self.data[f1][1]
            d1=d1[self.regs]
            f2 = np.random.choice(self.neighbors[f1])
            d2, y2 = self.data[f2][0], self.data[f2][1]
            d2=d2[self.regs]
            assert y1 == y2
            r = np.random.uniform(low=0, high=1)
            label = (y1)
            data = r*d1 + (1-r)*d2
            return (data, label)
    
    def __len__(self):
        return self.num_data


# Do Augmentation
    
flist = sub_name
aug_factor = 2
p_augmentation = True
num_neighbs = 5
lim4sim = 2
batch_size = 5
verbose = True
sim_function = functools.partial(cal_similarity, lim=lim4sim)

# For AAL
n_aal = int(num_edge_aal/4)

regions_inds_aal = get_regs(sub_name, n_aal, train_corr_aal)

training_set_aal = AugDataset(data=train_corr_aal, samples_list=sub_name, 
                                    augmentation=p_augmentation, aug_factor=aug_factor,
                                    num_neighbs=num_neighbs, eig_data=eig_data_aal, 
                                    similarity_fn=sim_function, 
                                    verbose=verbose,regs=regions_inds_aal)


# For HO
n_ho = int(num_edge_ho/4)

regions_inds_ho = get_regs(sub_name, n_ho, train_corr_ho)

training_set_ho = AugDataset(data=train_corr_ho, samples_list=sub_name, 
                                    augmentation=p_augmentation, aug_factor=aug_factor,
                                    num_neighbs=num_neighbs, eig_data=eig_data_ho, 
                                    similarity_fn=sim_function, 
                                    verbose=verbose,regs=regions_inds_ho)

# For CC
n_cc = int(num_edge_cc/4)

regions_inds_cc = get_regs(sub_name, n_cc, train_corr_cc)

training_set_cc = AugDataset(data=train_corr_cc, samples_list=sub_name, 
                                    augmentation=p_augmentation, aug_factor=aug_factor,
                                    num_neighbs=num_neighbs, eig_data=eig_data_cc, 
                                    similarity_fn=sim_function, 
                                    verbose=verbose,regs=regions_inds_cc)

# For Concatenated Data
n_all = (num_edge_aal + num_edge_ho + num_edge_cc)//4

regions_inds_all = get_regs(sub_name[11:], n_all, train_corr_all)

training_set_all = AugDataset(data=train_corr_all, samples_list=sub_name[11:], 
                                    augmentation=p_augmentation, aug_factor=aug_factor,
                                    num_neighbs=num_neighbs, eig_data=eig_data_all, 
                                    similarity_fn=sim_function, 
                                    verbose=verbose,regs=regions_inds_all)


# Prepare training data for AE
train_fc_aal = np.array([training_set_aal[f][0] for f in range(400)])
train_aal_label = np.array([training_set_aal[f][1] for f in range(400)])
train_fc_ho = np.array([training_set_ho[f][0] for f in range(400)])
train_ho_label = np.array([training_set_ho[f][1] for f in range(400)])
train_fc_cc = np.array([training_set_cc[f][0] for f in range(400)])
train_cc_label = np.array([training_set_cc[f][1] for f in range(400)]) # all three labels are the same

train_fc_all = np.array([training_set_all[f][0] for f in range(len(training_set_all))])
train_fc_label = np.array([training_set_all[f][1] for f in range(len(training_set_all))])

# Define Autoencoder in Keras
class Autoencoder(object):
    
    def __init__(self, inout_dim, encoded_dim):
        input_layer = Input(shape=(inout_dim,))
        hidden_input = Input(shape=(encoded_dim,))
        hidden_layer = Dense(encoded_dim, activation='relu')(input_layer)
        output_layer = Dense(inout_dim, activation='sigmoid')(hidden_layer)
        
        self._autoencoder_model = Model(input_layer, output_layer)
        self._encoder_model = Model(input_layer, hidden_layer)
        tmp_decoder_layer = self._autoencoder_model.layers[-1]
        self._decoder_model = Model(hidden_input, tmp_decoder_layer(hidden_input))
        
        self._autoencoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    def train(self, input_train, input_test, batch_size, epochs):
        self._autoencoder_model.fit(input_train,
                                    input_train,
                                    epochs = epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    validation_data=(
                                                     input_test,
                                                     input_test))
    def getEncodedImage(self, image):
        encoded_image = self._encoder_model.predict(image)
        return encoded_image
    
    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._decoder_model.predict(encoded_imgs)
        return decoded_image


autoencoder_aal = Autoencoder(train_fc_aal.shape[1], 2000)
autoencoder_aal.train(train_fc_aal, validation_edges_aal[regions_inds_aal,:].T, 30, 20)
encoded_train_edges_aal = autoencoder_aal.getEncodedImage(train_fc_aal)

autoencoder_ho = Autoencoder(train_fc_ho.shape[1], 2000)
autoencoder_ho.train(train_fc_ho, validation_edges_ho[regions_inds_ho,:].T, 30, 20)
encoded_train_edges_ho = autoencoder_ho.getEncodedImage(train_fc_ho)

autoencoder_cc = Autoencoder(train_fc_cc.shape[1], 2000)
autoencoder_cc.train(train_fc_cc, validation_edges_cc[regions_inds_cc,:].T, 30, 20)
encoded_train_edges_cc = autoencoder_cc.getEncodedImage(train_fc_cc)

autoencoder_all = Autoencoder(train_fc_all.shape[1], 2000)
autoencoder_all.train(train_fc_all, validation_edges[regions_inds_all,:].T, 30, 20)
encoded_train_edges_all = autoencoder_all.getEncodedImage(train_fc_all)

# Model on the separately concatenated data
encoded_augmentation_train_edges_separately = np.concatenate((encoded_train_edges_aal, encoded_train_edges_ho, encoded_train_edges_cc), axis = 1)


pipe_steps = [('scalar', StandardScaler()), ("clf", SVC(kernel = "rbf", probability=True))]
check_parameters = {
    'clf__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
}
pipeline = Pipeline(pipe_steps)
augmentation_auto_svm_separate_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
augmentation_auto_svm_separate_grid.fit(encoded_augmentation_train_edges_separately, train_aal_label)


encoded_validation_edges_aal = autoencoder_aal.getEncodedImage(validation_edges_aal[regions_inds_aal,:].T)
encoded_validation_edges_ho = autoencoder_ho.getEncodedImage(validation_edges_ho[regions_inds_ho,:].T)
encoded_validation_edges_cc = autoencoder_cc.getEncodedImage(validation_edges_cc[regions_inds_cc,:].T)
encoded_augmentation_validation_edges_separately = np.concatenate((encoded_validation_edges_aal, encoded_validation_edges_ho, encoded_validation_edges_cc), axis = 1)

y_predict = augmentation_auto_svm_separate_grid.predict(encoded_augmentation_validation_edges_separately)
accuracy = accuracy_score(validation_label,y_predict) ## Prediction accuracy
prob_predict = augmentation_auto_svm_separate_grid.predict_proba(encoded_augmentation_validation_edges_separately)
prob_predict = prob_predict.max(axis = 1)
fpr, tpr, thresholds = metrics.roc_curve(np.array(validation_label), np.array(prob_predict))
metrics.auc(fpr, tpr) ## AUC values
cmatrix = metrics.confusion_matrix(validation_label, y_predict) ## confusion matrix
sens = cmatrix[1, 1] / (cmatrix[1, 0] + cmatrix[1, 1])
spec = cmatrix[0, 0] / (cmatrix[0, 0] + cmatrix[0, 1])
print(metrics.classification_report(validation_label, y_predict))



# Model on the concatenated data
augmentation_auto_svm_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
augmentation_auto_svm_grid.fit(encoded_train_edges_all, train_fc_label)

encoded_augmentation_validation_edges = autoencoder_all.getEncodedImage(validation_edges[regions_inds_all,:].T)

y_predict = augmentation_auto_svm_grid.predict(encoded_augmentation_validation_edges)
accuracy = accuracy_score(validation_label,y_predict) ## Prediction accuracy
prob_predict = augmentation_auto_svm_grid.predict_proba(encoded_augmentation_validation_edges)
prob_predict = prob_predict.max(axis = 1)

validation_label_num = []
for v_lab in range(len(validation_label)):
    validation_label_num.append(1 if validation_label[v_lab] == 'ADHD' else 0) 

fpr, tpr, thresholds = metrics.roc_curve(np.array(validation_label_num), np.array(prob_predict))
metrics.auc(fpr, tpr) ## AUC values
cmatrix = metrics.confusion_matrix(validation_label, y_predict) ## confusion matrix
sens = cmatrix[1, 1] / (cmatrix[1, 0] + cmatrix[1, 1])
spec = cmatrix[0, 0] / (cmatrix[0, 0] + cmatrix[0, 1])
print(metrics.classification_report(validation_label, y_predict))
