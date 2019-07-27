#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')

train_data_80 = pd.read_csv('data_80_train.txt', sep = '\t')
test_data_80 = pd.read_csv('data_80_test.txt', sep = '\t')
train_data_2000 = pd.read_csv('data_2000_train.txt', sep = '\t')
test_data_2000 = pd.read_csv('data_2000_test.txt', sep = '\t')

combine = pd.concat([train_data_80, test_data_80])


# ## EDA

# In[ ]:


print('train_data_80 is of size:', train_data_80.shape)
print('test_data_80 is of size:', test_data_80.shape)
print('train_data_2000 is of size:', train_data_2000.shape)
print('test_data_2000 is of size:', test_data_2000.shape)
print('entire data is of size:', combine.shape)


# In[29]:


combine.head()


# In[30]:


combine.dtypes


# In[31]:


total = combine.isnull().sum(axis=0).sort_values(ascending = False)
percentage = (combine.isnull().sum(axis=0)/combine.count(axis=0)*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])
missing_data.head()


# In[32]:


print (combine['genre'].value_counts(),'\n')
print (combine['genre'].value_counts(normalize=True))


# In[33]:


# Create a correlation matrix for the combined data 
corr_metrics = combine.corr()
corr_metrics.style.background_gradient()


# In[16]:


sns.pairplot(combine.drop(['track_id']), hue="genre")
plt.show()


# In[38]:


plt.figure(figsize=(12, 12))
for i, feature in enumerate(combine.columns[1:-1]):
    plt.subplot(3,3,i+1)
    sns.distplot(combine[feature], hist=False)  
plt.show()


# In[8]:


combine.drop(['track_id']).groupby(['genre']).agg(['mean'])


# ## Feature Engineering

# In[2]:


# split features and genre from train and test datasets
train_features_80 = train_data_80.drop(['genre', 'track_id'], axis = 1)
train_labels_80 = train_data_80['genre']
test_features_80 = test_data_80.drop(['genre', 'track_id'], axis = 1)
test_labels_80 = test_data_80['genre']

train_features_2000 = train_data_2000.drop(['genre', 'track_id'], axis = 1)
train_labels_2000 = train_data_2000['genre']
test_features_2000 = test_data_2000.drop(['genre', 'track_id'], axis = 1)
test_labels_2000 = test_data_2000['genre']

combine_features = combine.drop(['genre', 'track_id'], axis = 1)

# Standardize the features
scaler = StandardScaler()
scaled_combine_features = scaler.fit_transform(combine_features)
scaled_train_features_80 = scaler.fit_transform(train_features_80)
scaled_test_features_80 = scaler.fit_transform(test_features_80)

scaled_train_features_2000 = scaler.fit_transform(train_features_2000)
scaled_test_features_2000 = scaler.fit_transform(test_features_2000)


# In[35]:


# This is just to make plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_combine_features)
exp_variance = pca.explained_variance_ratio_
n_components = pca.n_components_
print(exp_variance)
print(n_components)


# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar([x+1 for x in range(n_components)], exp_variance)
ax.set_xlabel('Principal Component #')
ax.set_ylabel('explained variance ratio')


# In[36]:


# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.9.
fig, ax = plt.subplots()
ax.plot([x+1 for x in range(n_components)], cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
ax.set_xlabel('Principal Component #')
ax.set_ylabel('cumulative explained variance ratio')


# In[3]:


# Perform PCA with the chosen number of components and project data onto components
n_components = 6
pca = PCA(n_components, random_state=20)

pca_fit_train_80 = pca.fit_transform(scaled_train_features_80)
pca_fit_test_80 = pca.fit_transform(scaled_test_features_80)

pca_fit_train_2000 = pca.fit_transform(scaled_train_features_2000)
pca_fit_test_2000 = pca.fit_transform(scaled_test_features_2000)


# ## Baseline Models

# ## 1.multinomial logistic regression

# Weighted technique:

# In[5]:


# Train our logistic regression and predict labels for the test set
# Create one-vs-rest logistic regression object
logreg = LogisticRegression(multi_class='multinomial', solver='newton-cg', class_weight = 'balanced')
# Import the model we are using

# pca data on 80:20
logreg.fit(pca_fit_train_80, train_labels_80)
print("Logistic Regression Classification test accuracy with PCA 80:20 \n", classification_report(test_labels_80, logreg.predict(pca_fit_test_80)))

# data on 80:20
logreg.fit(scaled_train_features_80, train_labels_80)
print("Logistic Regression Classification test accuracy with scaled 80:20 \n", classification_report(test_labels_80, logreg.predict(scaled_test_features_80)))


# Under Sampling Technique:

# In[40]:


# pca data on 2000
logreg.fit(pca_fit_train_2000, train_labels_2000)
print("Logistic Regression Classification test accuracy with PCA 2000 \n", classification_report(test_labels_2000, logreg.predict(pca_fit_test_2000)))

# data on 2000
logreg.fit(train_features_2000, train_labels_2000)
print("Logistic Regression Classification test accuracy with 2000 \n", classification_report(test_labels_2000, logreg.predict(test_features_2000)))


# Over Sampling Technique:

# In[6]:


ros = RandomOverSampler(random_state=42)
train_features_over, train_labels_over = ros.fit_resample(scaled_train_features_80, train_labels_80)

# data on 2000
logreg.fit(train_features_over, train_labels_over)
print("Logistic Regression Classification test accuracy with scaled 80:20 \n", classification_report(test_labels_80, logreg.predict(scaled_test_features_80)))


# The multinomial logistic regression model performed better on raw data than PCA imposed data.

# ## 2.random forest

# Random Forest without sampling and weighted technique:

# In[ ]:


rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 250, class_weight = 'balanced')
rf.fit(pca_fit_train_80, train_labels_80)
#print("Random Forest Classification train accuracy with raw 1000 ", metrics.accuracy_score(train_labels_1000, rf.predict(train_features_1000)))
print("Random Forest Classification test accuracy ", classification_report(test_labels_80, rf.predict(pca_fit_test_80)))

rf.fit(train_features_80, train_labels_80)
#print("Random Forest Classification train accuracy with raw 1000 ", metrics.accuracy_score(train_labels_1000, rf.predict(train_features_1000)))
print("Random Forest Classification test accuracy ", classification_report(test_labels_80, rf.predict(test_features_80)))


# Random Forest with undersampling technique:

# In[ ]:


rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 250)
rf.fit(scaled_train_features_2000, train_labels_2000)
#print("Random Forest Classification train accuracy with raw 1000 ", metrics.accuracy_score(train_labels_1000, rf.predict(train_features_1000)))
print("Random Forest Classification test accuracy ", classification_report(test_labels_2000, rf.predict(test_features_2000)))


# Random Forest with oversampling technique:

# In[ ]:


rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 250)
rf.fit(train_features_over, train_labels_over)
#print("Random Forest Classification train accuracy with raw 1000 ", metrics.accuracy_score(train_labels_1000, rf.predict(train_features_1000)))
print("Random Forest Classification test accuracy ", classification_report(test_labels_80, rf.predict(test_features_80)))


# In[12]:


rf = RandomForestClassifier()

rf.fit(pca_fit_train_80, train_labels_80)
print('random forest on PCA 80: \n', classification_report(test_labels_80, rf.predict(pca_fit_test_80)))

rf.fit(train_features_80, train_labels_80)
print('random forest on 80: \n', classification_report(test_labels_80, rf.predict(test_features_80)))

rf.fit(pca_fit_train_2000, train_labels_2000)
print('random forest on PCA 2000: \n', classification_report(test_labels_2000, rf.predict(pca_fit_test_2000)))

rf.fit(train_features_2000, train_labels_2000)
print('random forest on 2000: \n', classification_report(test_labels_2000, rf.predict(test_features_2000)))


# Random forest performed better on raw data than PCA imposed data.

# ## XGBoost (without any sampling and unweighted)

# In[4]:


xgb = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgb.fit(train_features_80, train_labels_80)
predictions = xgb.predict(test_features_80)
print('XGBoost accuracy rate on 80 data: \n', classification_report(test_labels_80, predictions))

xgb.fit(pca_fit_train_80, train_labels_80)
predictions = xgb.predict(pca_fit_test_80)
print('XGBoost accuracy rate on PCA 80 data: \n', classification_report(test_labels_80, predictions))

xgb.fit(train_features_2000, train_labels_2000)
predictions = xgb.predict(test_features_2000)
print('XGBoost accuracy rate on 2000 data: \n', classification_report(test_labels_2000, predictions))

xgb.fit(pca_fit_train_2000, train_labels_2000)
predictions = xgb.predict(pca_fit_test_2000)
print('XGBoost accuracy rate on PCA 2000 data: \n', classification_report(test_labels_2000, predictions))

