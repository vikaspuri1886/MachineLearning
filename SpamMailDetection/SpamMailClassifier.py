#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:22:39 2020

@author: vikaspuri
"""

import pandas as pd
import re
import nltk

messages = pd.read_csv("smsspamcollection/SMSSpamCollection", sep = "\t", names = ["label", "message"])

nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(0, len(messages)):
    review = re.sub("[^a-zA-Z]", " ", messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer()
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

y_pred  = spam_detect_model.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_mtrx = confusion_matrix(y_test, y_pred)

acc_score = accuracy_score(y_test, y_pred)