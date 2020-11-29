# -*- coding: utf-8 -*-
"""
This script performs tumor subtype classification from expert shape features.

Created on Sat Nov 28 16:07:27 2020

@author: reddyav1
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

features_manual = np.load('Data/shape_features/feat_manual.npy')
labels = np.load('Data/shape_features/labels.npy')

k = 3
num_feat = np.size(features_manual,1)

# Visualizing the data
plt.close("all")
class_features = [features_manual[labels == c] for c in range(3)]

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')

ax.scatter(class_features[0][:,0],class_features[0][:,1],class_features[0][:,2], c='r', marker='o')
ax.scatter(class_features[1][:,0],class_features[1][:,1],class_features[1][:,2], c='b', marker='o')
ax.scatter(class_features[2][:,0],class_features[2][:,1],class_features[2][:,2], c='g', marker='o')
ax.set_xlabel('Angular Standard Deviation')
ax.set_ylabel('Margin Fluctuation')
ax.set_zlabel('BEVR')

fig2 = plt.figure()
ax = fig2.add_subplot(1,3,1)
ax.scatter(class_features[0][:,0],class_features[0][:,1])
ax.scatter(class_features[1][:,0],class_features[1][:,1])
ax.scatter(class_features[2][:,0],class_features[2][:,1])
ax.set_xlabel('Angular Standard Deviation')
ax.set_ylabel('Margin Fluctuation')

ax = fig2.add_subplot(1,3,2)
ax.scatter(class_features[0][:,0],class_features[0][:,2])
ax.scatter(class_features[1][:,0],class_features[1][:,2])
ax.scatter(class_features[2][:,0],class_features[2][:,2])   
ax.set_xlabel('Angular Standard Deviation')
ax.set_ylabel('BEVR')


ax = fig2.add_subplot(1,3,3)
ax.scatter(class_features[0][:,2],class_features[0][:,1])
ax.scatter(class_features[1][:,2],class_features[1][:,1])
ax.scatter(class_features[2][:,2],class_features[2][:,1])   
ax.set_xlabel('BEVR')
ax.set_ylabel('Margin Fluctuation')

# Classification
classifiers = [KNeighborsClassifier(3),
    LogisticRegression(),
    GaussianNB(),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="rbf",gamma=2, C=1),]

rand_seed = 10;
spl1 = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=rand_seed)

classifier_accuracies = []

for i in range(len(classifiers)):
    clf = classifiers[i]
    fold_accuracies = []
    for train_index, test_index in spl1.split(features_manual,labels):    
        # Split up data into test and train data
        X_train, X_test = features_manual[train_index], features_manual[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Normalization can help accuracy 
        scl = MinMaxScaler() 
        X_train = scl.fit_transform(X_train)  
        X_test = scl.transform(X_test)
    
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fold_accuracies.append(np.sum(y_pred == y_test)/np.size(y_test,0))
    
    kfold_accuracy = np.mean(fold_accuracies)
    classifier_accuracies.append(kfold_accuracy)


clf_names = ['KNN (K=3)', 'Logistic Reg.', 'Naive Bayes', 'SVM Linear', 'SVM RBF']

x_pos = [i for i, _ in enumerate(clf_names)]

plt.figure()
plt.style.use('ggplot')
plt.grid(True)
plt.barh(x_pos, classifier_accuracies, color='green')
plt.ylabel("Classifier")
plt.xlabel("Accuracy")
plt.title("CoC Classification Accuracy")
plt.xlim(0,1)
plt.yticks(x_pos, clf_names)


plt.show()