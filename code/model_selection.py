#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:28:50 2018

@author: wsd
"""
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors  
from sklearn.linear_model import Lasso  
from sklearn.linear_model import LassoCV  
import re
from scipy.sparse import hstack


cleanedData = pd.read_csv('new_train.csv')

cleanedData['clean']=[text+" shenmegui" for text in cleanedData['clean']]
cleanedData['categories']=[re.sub('[^a-zA-Z]',' ', cate) for cate in cleanedData['categories']]
cleanedData['name']=[name.lower() for name in cleanedData['name']]
cleanedData['city']=[city.lower() for city in cleanedData['city']]
name_cate=cleanedData['name']+" "+cleanedData['categories']+" "+cleanedData['city']
name_cate_cv = CountVectorizer(strip_accents='ascii', analyzer='word', binary=True)
name_cate_fit=name_cate_cv.fit_transform(name_cate.iloc[range(100)])
test_name_cate_fit=name_cate_cv.transform(name_cate.iloc[range(1540000,1540100)])

cleanedText=cleanedData.loc[range(100),'clean']
cleanedLabel=cleanedData.loc[range(100),'stars']
cleanedText_test=cleanedData.loc[range(1540000,1540100),'clean']
cleanedLabel_test=cleanedData.loc[range(1540000,1540100),'stars']

#trasfer test text
validateData = pd.read_csv('new_test.csv')
validateData['clean']=[text+" shenmegui" for text in validateData['clean']]
validateData['categories']=[re.sub('[^a-zA-Z]',' ', cate) for cate in validateData['categories']]
validateData['name']=[name.lower() for name in validateData['name']]
validateData['city']=[city.lower() for city in validateData['city']]
vali_name_cate=validateData['name']+" "+validateData['categories']+" "+validateData['city']
vali_name_cate_fit=name_cate_cv.transform(vali_name_cate)
cleanedText_vali=validateData['clean']

def Bitoken(text,n=100,score_fn=BigramAssocMeasures.chi_sq):
    WT=word_tokenize(text)
    bigram_finder = BigramCollocationFinder.from_words(WT) #把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)
    biList=[' '.join(pair) for pair in bigrams]
    return WT+biList

#####################################################
#based on TF-IDF
tf = TfidfVectorizer(strip_accents='ascii', analyzer='word',tokenizer=Bitoken)
tf_fit = tf.fit_transform(cleanedText)
test_fit=tf.transform(cleanedText_test)
vali_fit=tf.transform(cleanedText_vali)
from sklearn.externals import joblib
joblib.dump(tf,'tf.pkl')  #tf=joblib.load('tf.pkl')
joblib.dump(tf_fit,'tf_fit.pkl')
joblib.dump(test_fit,'test_fit.pkl')
joblib.dump(vali_fit,'vali_fit.pkl')
'''
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
'''
train_mat=hstack([name_cate_fit,tf_fit]);joblib.dump(train_mat,'train_mat.pkl')
test_mat=hstack([test_name_cate_fit,test_fit]);joblib.dump(test_mat,'test_mat.pkl')
vali_mat=hstack([vali_name_cate_fit,vali_fit]);joblib.dump(vali_mat,'vali_mat.pkl')
train_mat=joblib.load('train_mat.pkl')
test_mat=joblib.load('test_mat.pkl')
vali_mat=joblib.load('vali_mat.pkl')

#SVM classifier
classifier_tf = LinearSVC()   #kernel='rbf'
classifier_tf.fit(train_mat, cleanedLabel)
pre_tf = classifier_tf.predict(test_mat)
MSE_SVM=sum((np.array(cleanedLabel_test) -pre_tf)**2)/cleanedLabel_test.shape[0]
MSE_SVM
pre_vali_svm=classifier_tf.predict(vali_mat)
df=pd.DataFrame(data=pre_vali_svm)
df.to_csv('svm.csv', sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)

#SVM regression
svr=SVR()
svr.fit(train_mat,cleanedLabel)
pre = svr.predict(test_mat)
MSE_SVR=sum((np.array(cleanedLabel_test) -pre)**2)/cleanedLabel_test.shape[0]

#LASSO
lassocv_tf = LassoCV()
lassocv_tf.fit(train_mat, cleanedLabel)
alpha = lassocv_tf.alpha_

lasso_tf = Lasso()#alpha=0.5)#alpha)
lasso_tf.fit(train_mat, cleanedLabel)
pre_tf=lasso_tf.predict(test_mat)
MSE_LASSO=sum((np.array(cleanedLabel_test) - pre_tf)**2)/cleanedLabel_test.shape[0]
joblib.dump(lasso_tf,'lasso_tf.pkl')
pre_vali_lasso=lasso_tf.predict(vali_mat)
joblib.dump(pre_vali_lasso,'pre_vali_lasso.pkl')

#Logistic
#from sklearn.linear_model import LogisticRegression
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(tf_fit)
X_train_std = sc.transform(tf_fit)
X_test_std = sc.transform(test_fit)
lr = LogisticRegression()
lr.fit(X_train_std, cleanedLabel)
lr.predict_proba(X_test_std[0,:])   ## 查看第一个测试样本属于各个类别的概率
pre_tf=lr.predict(test_fit)
MSE_LOGI=sum((np.array(cleanedLabel_test) - pre_tf)**2)/cleanedLabel_test.shape[0]
'''

pre1=pd.read_csv('pre1.csv')
pre2=pd.read_csv('pre2.csv')
pre3=pd.read_csv('pre3.csv')
train_mat=hstack([pre1['pre1'],train_mat])
test_mat=hstack([pre2['pre2'],test_mat])
vali_mat=hstack([pre3['pre3'],vali_mat])
lr = LogisticRegression()
lr.fit(train_mat, cleanedLabel)
prediction=lr.predict_proba(test_mat)
pre_tf=np.dot(lr.predict_proba(test_mat),np.array([[1],[2],[3],[4],[5]]))
#pre_tf=lr.predict(test_mat)
MSE_LOGI=sum((np.array(cleanedLabel_test) - pre_tf)**2)/cleanedLabel_test.shape[0]
vali_pre=np.dot(lr.predict_proba(vali_mat),np.array([[1],[2],[3],[4],[5]]))
df=pd.DataFrame(data=vali_pre)
df.to_csv('logi_LSTM.csv', sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)





#####################################################
#based on counts
cv = CountVectorizer(strip_accents='ascii', analyzer='word',tokenizer=Bitoken)
cv_fit=cv.fit_transform(cleanedText)
test_cv_fit=cv.transform(cleanedText_test)

#SVM
classifier_cv = LinearSVC()
classifier_cv.fit(cv_fit, cleanedLabel)
pre_cv = classifier_cv.predict(test_cv_fit)
sum((np.array(cleanedLabel_test) - pre_cv)**2)/200000

#random forest
rf_cv = RandomForestRegressor()
rf_cv.fit(cv_fit,cleanedLabel)
pre_cv = rf_cv.predict(test_fit)
sum((np.array(cleanedLabel_test) - pre_cv)**2)/200000

#KNN
clf_cv = neighbors.KNeighborsClassifier(n_neighbors = 20,
                                     algorithm = 'kd_tree', weights='uniform')  
clf_cv.fit(cv_fit, cleanedLabel)
pre_cv = clf_cv.predict(test_fit)
sum((np.array(cleanedLabel_test) - pre_cv)**2)/200000



