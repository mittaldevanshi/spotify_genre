#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from sklearn import metrics
from IPython.display import Image  
import pydotplus
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


train_data_80 = pd.read_csv('data_80_train.txt', sep = '\t')
test_data_80 = pd.read_csv('data_80_test.txt', sep = '\t')
train_data_1000 = pd.read_csv('data_1000_train.txt', sep = '\t')
test_data_1000 = pd.read_csv('data_1000_test.txt', sep = '\t')
train_data_2000 = pd.read_csv('data_2000_train.txt', sep = '\t')
test_data_2000 = pd.read_csv('data_2000_test.txt', sep = '\t')

combine_80 = pd.concat([train_data_80, test_data_80])
combine_1000 = pd.concat([train_data_1000, test_data_1000])
combine_2000 = pd.concat([train_data_2000, test_data_2000])

# Create a correlation matrix for the combined data 
corr_metrics = combine_80.corr()
corr_metrics.style.background_gradient()

#variables loudness and energy are strongly correlated -- hence, we pemodelorm PCA to perform feature engineering


#Creating train and test datasets with features and labels

train_features_80 = train_data_80.drop(['genre', 'track_id'], axis = 1)
train_labels_80 = train_data_80['genre']

test_features_80 = test_data_80.drop(['genre', 'track_id'], axis = 1)
test_labels_80 = test_data_80['genre']
combine_features_80 = combine_80.drop(['genre', 'track_id'], axis = 1)

train_features_1000 = train_data_1000.drop(['genre', 'track_id'], axis = 1)
train_labels_1000 = train_data_1000['genre']

test_features_1000 = test_data_1000.drop(['genre', 'track_id'], axis = 1)
test_labels_1000 = test_data_1000['genre']
combine_features_1000 = combine_1000.drop(['genre', 'track_id'], axis = 1)

train_features_2000 = train_data_2000.drop(['genre', 'track_id'], axis = 1)
train_labels_2000 = train_data_2000['genre']

test_features_2000 = test_data_2000.drop(['genre', 'track_id'], axis = 1)
test_labels_2000 = test_data_2000['genre']
combine_features_2000 = combine_2000.drop(['genre', 'track_id'], axis = 1)

# Scale the features and set the values to a new variable
scaler = StandardScaler()
scaled_combine_features_80 = scaler.fit_transform(combine_features_80)
scaled_train_features_80 = scaler.fit_transform(train_features_80)
scaled_test_features_80 = scaler.fit_transform(test_features_80)

scaled_combine_features_1000 = scaler.fit_transform(combine_features_1000)
scaled_train_features_1000 = scaler.fit_transform(train_features_1000)
scaled_test_features_1000 = scaler.fit_transform(test_features_1000)

scaled_combine_features_2000 = scaler.fit_transform(combine_features_2000)
scaled_train_features_2000 = scaler.fit_transform(train_features_2000)
scaled_test_features_2000 = scaler.fit_transform(test_features_2000)

# Normalize the entire data column
normalized_train_80 = preprocessing.normalize(train_features_80)
#train_features_normalized = pd.DataFrame(normalized_X)
normalized_test_80 = preprocessing.normalize(test_features_80)
#test_features_normalized = pd.DataFrame(normalized_test)

normalized_train_1000 = preprocessing.normalize(train_features_1000)
#train_features_normalized = pd.DataFrame(normalized_X)
normalized_test_1000 = preprocessing.normalize(test_features_1000)
#test_features_normalized = pd.DataFrame(normalized_test)

normalized_train_2000 = preprocessing.normalize(train_features_2000)
#train_features_normalized = pd.DataFrame(normalized_X)
normalized_test_2000 = preprocessing.normalize(test_features_2000)
#test_features_normalized = pd.DataFrame(normalized_test)

n_components = 6
# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components, random_state=20)
pca.fit(scaled_combine_features_80)

pca_fit_train_80 = pca.fit_transform(scaled_train_features_80)
pca_fit_test_80 = pca.fit_transform(scaled_test_features_80)

pca_fit_train_1000 = pca.fit_transform(scaled_train_features_1000)
pca_fit_test_1000 = pca.fit_transform(scaled_test_features_1000)

pca_fit_train_2000 = pca.fit_transform(scaled_train_features_2000)
pca_fit_test_2000 = pca.fit_transform(scaled_test_features_2000)


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
train_features_over, train_labels_over = ros.fit_resample(scaled_train_features_80, train_labels_80)

# scaled data on 80:20
xgb1 = XGBClassifier(
 learning_rate =0.001,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgb1.fit(train_features_over, train_labels_over)
print("Logistic Regression Classification test accuracy with 80:20 ", classification_report(test_labels_80, xgb1.predict(scaled_test_features_80)))

# ## Random Forest without sampling and weighted

rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 250, class_weight = 'balanced')
rf.fit(pca_fit_train_80, train_labels_80)
#print("Random Forest Classification train accuracy with raw 1000 ", metrics.accuracy_score(train_labels_1000, rf.predict(train_features_1000)))
print("Random Forest Classification test accuracy ", classification_report(test_labels_80, rf.predict(pca_fit_test_80)))

rf.fit(train_features_80, train_labels_80)
#print("Random Forest Classification train accuracy with raw 1000 ", metrics.accuracy_score(train_labels_1000, rf.predict(train_features_1000)))
print("Random Forest Classification test accuracy ", classification_report(test_labels_80, rf.predict(test_features_80)))

# ## Random Forest with undersampling

rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 250)
rf.fit(scaled_train_features_2000, train_labels_2000)
#print("Random Forest Classification train accuracy with raw 1000 ", metrics.accuracy_score(train_labels_1000, rf.predict(train_features_1000)))
print("Random Forest Classification test accuracy ", classification_report(test_labels_2000, rf.predict(test_features_2000)))

# ## Random Forest with oversampling

rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 250)
rf.fit(train_features_over, train_labels_over)
#print("Random Forest Classification train accuracy with raw 1000 ", metrics.accuracy_score(train_labels_1000, rf.predict(train_features_1000)))
print("Random Forest Classification test accuracy ", classification_report(test_labels_80, rf.predict(test_features_80)))

