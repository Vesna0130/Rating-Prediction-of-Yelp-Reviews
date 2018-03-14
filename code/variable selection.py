#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:46:15 2018

@author: wsd
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from scipy.sparse import hstack
uni=pd.read_csv('words_uni.csv')
bi=pd.read_csv('words_bi.csv')
tri=pd.read_csv('words_tri.csv')

uni_10000=uni.iloc[range(10000),:]
bi_20000=bi.iloc[range(20000),:]
tri_50000=tri.iloc[range(50000),:]

uni_10000.to_csv('uni_10000.csv', sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)
bi_20000.to_csv('bi_20000.csv', sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)
tri_50000.to_csv('tri_50000.csv', sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)


cleanedData = pd.read_csv('new_train.csv')
uni_cv=CountVectorizer(strip_accents='ascii', analyzer='word', binary=True)
uni_mat =  uni_cv.fit_transform(cleanedData['clean'])

#cleanedData['stars']
uni_star_mat=uni_mat.toarray()
uni_mat_1000=uni_mat.toarray()
#uni_star=hstack([uni_mat,list(cleanedData['stars'])])
for i in range(uni_mat.shape[0]):
    if i % 100 ==0:
        print(i)
    uni_star_mat[i,:]=uni_mat_1000[i,:]*cleanedData['stars'][i]
    
var=np.zeros([uni_star_mat.shape[1]])
for i in range(uni_star_mat.shape[0]):
    var[i]=np.var(uni_star_mat[:,i])

words=uni_cv.get_feature_names()
df=pd.DataFrame(data={'words':words,'variance':var})
df=df.sort_values('variance',ascending=True)
df.to_csv('uni_word_var.csv', sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)

