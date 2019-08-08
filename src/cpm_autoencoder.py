#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:41:35 2019

@author: Wei Dai 
"""


from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFdr, SelectPercentile
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc
from scipy import stats

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

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


# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Concatenate three datasets together, use AE for feature selection and SVM for prediction

# Keras implementation
autoencoder = Autoencoder(train_edges.shape[0], 10000)
autoencoder.train(train_edges.T, validation_edges.T, 50, 20)

encoded_train_edges = autoencoder.getEncodedImage(train_edges.T)
decoded_train_edges = autoencoder.getDecodedImage(encoded_train_edges)


# Use encoded training data as selected data to train the model
pipe_steps = [('scalar', StandardScaler()), ("clf", SVC(kernel = "rbf", probability=True))]
check_parameters = {
    'clf__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
}
pipeline = Pipeline(pipe_steps)
auto_svm_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
auto_svm_grid.fit(encoded_train_edges, train_label)

encoded_validation_edges = autoencoder.getEncodedImage(validation_edges.T)
y_predict = auto_svm_grid.predict(encoded_validation_edges)
accuracy = accuracy_score(validation_label,y_predict) ## Prediction accuracy
prob_predict = auto_svm_grid.predict_proba(encoded_validation_edges)
prob_predict = prob_predict.max(axis = 1)
fpr, tpr, thresholds = metrics.roc_curve(np.array(validation_label), np.array(prob_predict))
metrics.auc(fpr, tpr) ## AUC values
cmatrix = metrics.confusion_matrix(validation_label, y_predict) ## confusion matrix
sens = cmatrix[1, 1] / (cmatrix[1, 0] + cmatrix[1, 1])
spec = cmatrix[0, 0] / (cmatrix[0, 0] + cmatrix[0, 1])
print(metrics.classification_report(validation_label, y_predict))


# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# Autocoder select features separately and then concatenate together for prediction in SVM

autoencoder_aal = Autoencoder(train_edges_aal.shape[0], 2000)
autoencoder_aal.train(train_edges_aal.T, validation_edges_aal.T, 10, 20)
encoded_train_edges_aal = autoencoder_aal.getEncodedImage(train_edges_aal.T)

autoencoder_ho = Autoencoder(train_edges_ho.shape[0], 2000)
autoencoder_ho.train(train_edges_ho.T, validation_edges_ho.T, 10, 20)
encoded_train_edges_ho = autoencoder_ho.getEncodedImage(train_edges_ho.T)

autoencoder_cc = Autoencoder(train_edges_cc.shape[0], 2000)
autoencoder_cc.train(train_edges_cc.T, validation_edges_cc.T, 10, 20)
encoded_train_edges_cc = autoencoder_cc.getEncodedImage(train_edges_cc.T)

encoded_train_edges_separately = np.concatenate((encoded_train_edges_aal, encoded_train_edges_ho, encoded_train_edges_cc), axis = 1)


pipe_steps = [('scalar', StandardScaler()), ("clf", SVC(kernel = "rbf", probability=True))]
check_parameters = {
    'clf__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
}
pipeline = Pipeline(pipe_steps)
auto_svm_separate_grid = GridSearchCV(pipeline, param_grid=check_parameters, cv=5)
auto_svm_separate_grid.fit(encoded_train_edges_separately, train_label)

# auto_svm_separate_grid.best_score_
# auto_svm_separate_grid.best_estimator_

encoded_validation_edges_aal = autoencoder_aal.getEncodedImage(validation_edges_aal.T)
encoded_validation_edges_ho = autoencoder_ho.getEncodedImage(validation_edges_ho.T)
encoded_validation_edges_cc = autoencoder_cc.getEncodedImage(validation_edges_cc.T)
encoded_validation_edges_separately = np.concatenate((encoded_validation_edges_aal, encoded_validation_edges_ho, encoded_validation_edges_cc), axis = 1)

y_predict = auto_svm_separate_grid.predict(encoded_validation_edges_separately)
accuracy = accuracy_score(validation_label,y_predict) ## Prediction accuracy
prob_predict = auto_svm_separate_grid.predict_proba(encoded_validation_edges_separately)
prob_predict = prob_predict.max(axis = 1)
fpr, tpr, thresholds = metrics.roc_curve(np.array(validation_label), np.array(prob_predict))
metrics.auc(fpr, tpr) ## AUC values
cmatrix = metrics.confusion_matrix(validation_label, y_predict) ## confusion matrix
sens = cmatrix[1, 1] / (cmatrix[1, 0] + cmatrix[1, 1])
spec = cmatrix[0, 0] / (cmatrix[0, 0] + cmatrix[0, 1])
print(metrics.classification_report(validation_label, y_predict))


# tSNE
train_embedded = TSNE(n_components=2, perplexity = 100).fit_transform(train_edges.T)
Xax=train_embedded[:,0]
Yax=train_embedded[:,1]
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
plt.xlabel("First Component",fontsize=14)
plt.ylabel("Second Component",fontsize=14)
plt.legend()
plt.show()

