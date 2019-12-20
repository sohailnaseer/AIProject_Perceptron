# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:05:33 2019

@author: Black_Death
"""

from sklearn.model_selection import train_test_split
import sys
import pandas as pd

dataset = pd.read_csv(r'{0}'.format(sys.argv[1]),header=None)
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.join(y_train)
X_train.to_csv('train_dataset.csv',header=None)

X_test = X_test.join(y_test)
X_test.to_csv('test_dataset.csv',header=None)