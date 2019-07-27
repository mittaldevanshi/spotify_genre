#!/usr/bin/env python
# coding: utf-8

# In[33]:


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
from IPython.display import Image  
import pydotplus
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('~/Downloads/data_80_train.txt', sep = '\t')
test_data = pd.read_csv('~/Downloads/data_80_test.txt', sep = '\t')

combine = pd.concat([train_data, test_data])

# Create a correlation matrix for the combined data 
corr_metrics = combine.corr()
corr_metrics.style.background_gradient()

#variables loudness and energy are strongly correlated -- hence, we perform PCA to perform feature engineering


# In[34]:


#Creating train and test datasets with features and labels

train_features = train_data.drop(['genre', 'track_id'], axis = 1)
train_labels = train_data['genre']

test_features = test_data.drop(['genre', 'track_id'], axis = 1)
test_labels = test_data['genre']
combine_features = combine.drop(['genre', 'track_id'], axis = 1)

#Normalizing our features

# Scale the features and set the values to a new variable
scaler = StandardScaler()
scaled_combine_features = scaler.fit_transform(combine_features)
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.fit_transform(test_features)


# In[35]:


# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_combine_features)
exp_variance = pca.explained_variance_ratio_

#print(combine_features.columns.values, exp_variance)
#print(pca.n_components_)

features = list(combine_features.columns.values)
fig, ax = plt.subplots()
ax.bar(range(9), exp_variance)
ax.set_xlabel('Principal Component #')
ax.set_ylabel("Variance Explained")
#ax.set_xticklabels(features)
fig = ax.get_figure()
fig.savefig('Variance of Components')


# In[36]:



# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90 since we are trying to find that 
#90% of the variance can be explained by these many components
fig, ax = plt.subplots()
ax.plot(range(9), cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
fig = ax.get_figure()
fig.savefig('PCA')

#we see 6 components that can explain 90% of the variance

n_components = 6
# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components, random_state=20)
pca.fit(scaled_combine_features)
pca_fit_train = pca.fit_transform(scaled_train_features)
pca_fit_test = pca.fit_transform(scaled_test_features)


# In[37]:


#Decision Tree
clf_gini = DecisionTreeClassifier(criterion = "entropy",
                                min_samples_leaf=30, max_depth = 20)
print(clf_gini.fit(train_features, train_labels))
prediction = clf_gini.predict(test_features)
#print(prediction)
print("Accuracy is ", accuracy_score(test_labels,prediction)*100)
#print(clf_gini.score(test_labels,prediction))
print("F1 score is", f1_score(test_labels,prediction, average='micro'))


# In[38]:


'''
feature_cols = ["Component 1","Component 2","Component 3","Component 4","Component 5", "Component 6"]
test_label_unique = (test_labels.unique())
dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=test_label_unique)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('pca_genres.png')
Image(graph.create_png())
'''


# In[ ]:


#print("The overall accuracy is", (sum(dtree_predictions == test_labels))/len(test_labels))
comparison = prediction == test_labels

combined = prediction
combine = pd.DataFrame(prediction)
combine['True values'] = test_labels
combine['Comparison'] = pd.DataFrame(comparison)
test_label_unique = (test_labels.unique())
#print(test_label_unique)

#print("The true and predicted values are", combine)
print(classification_report(test_labels, prediction))


# In[ ]:


# Train our logistic regression and predict labels for the test set
# Create one-vs-rest logistic regression object
logreg = LogisticRegression(random_state = 10, multi_class='multinomial', solver='newton-cg')

# unnormalized data
logreg.fit(train_features, train_labels)
print ("Logistic regression Train Accuracy with unnormalized: ", metrics.accuracy_score(train_labels, logreg.predict(train_features)))
print ("Logistic regression Test Accuracy with unnormalized: ", metrics.accuracy_score(test_labels, logreg.predict(test_features)))

# with pca_fitted data
logreg.fit(pca_fit_train, train_labels)
print ("Logistic regression Train Accuracy with PCA: ", metrics.accuracy_score(train_labels, logreg.predict(pca_fit_train)))
print ("Logistic regression Test Accuracy with PCA: ", metrics.accuracy_score(test_labels, logreg.predict(pca_fit_test)))

# without pca_fitted data
logreg.fit(scaled_train_features, train_labels)
print ("Logistic regression Train Accuracy without PCA: ", metrics.accuracy_score(train_labels, logreg.predict(scaled_train_features)))
print ("Logistic regression Test Accuracy without PCA: ", metrics.accuracy_score(test_labels, logreg.predict(scaled_test_features)))


# The multinomial logistic regression model performed better without PCA-transformed data than PCA imposed data.

# In[ ]:


logreg.fit(scaled_train_features, train_labels)
test_preds = logreg.predict(scaled_test_features)
confusion_matrix = confusion_matrix(test_labels, test_preds)
print(confusion_matrix)


# In[ ]:


comparison = (test_preds == test_labels)
combine = pd.DataFrame(pred_labels_logit, columns=['predictions'])
combine['True values'] = test_labels
combine['Comparison'] = pd.DataFrame(comparison)
print("The true and predicted values are \n", combine)
test_label_unique = (test_labels.unique())
class_rep_log = classification_report(test_labels, pred_labels_logit, target_names=test_label_unique)
print("Logistic Regression: \n", class_rep_log)


# In[ ]:


import seaborn as sns
class_names=test_label_unique # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

