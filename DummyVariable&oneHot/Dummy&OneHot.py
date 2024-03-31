# we work again with a predicti model for prices but now we have houses in different towns. 
# problem is town is described by charachters not numbers

# CATEGORICAL VARIABLES
# Nominal-> Not any numerical order between them (Male,Female)
# Ordinal-> Some numerical order between them ( High, Medium, Low)


# ONE HOT ENCODING
#  we will use ONE HOT ENCODING  to work with our nominal values
# create a code variable (dummy variable) for each city . For example for monore 100 and robinsvile 010
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.linear_model import LinearRegression
from word2number import w2n
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("C:\\Users\\kosta\\OneDrive - Πολυτεχνείο Κρήτης\\Μαθήματα\\10ο Εξάμηνο\\Ml_Python_Course\\DummyVariable&oneHot\\homeprices.csv")

# get dummy variables usind panda
dummies=pd.get_dummies(df.town)


# concatanate these dummies to the old df
merged= pd.concat([df,dummies],axis='columns')

# drop the original town column we dont need it anymore
merged= merged.drop(columns=['town'],axis='columns')


#  we now need to drop one dummy variable column in order to not get dummy variable trap
merged= merged.drop(columns=['west windsor'],axis='columns')

# crete the linear regression model
model= LinearRegression()

# now we have to train the model x= area ,monroe township ,robinsvile 
X = merged.drop('price',axis='columns')
#  Y= price
y = merged.price

# train 
model.fit(X,y)
# now predict value for a  robinsvile house
p=model.predict([[2800,0,1]])
# now predict value for a  monroe house

p2=model.predict([[3400,0,0]])

# check hoe accurate the model is
score=model.score(X,y)


#  SKLEARN ENCODER (do the same )
le = LabelEncoder()

dfle= df
#  return labeled column for town
dfle.town=le.fit_transform(dfle.town)

# this time we use values in order for X to be a 2D area and not a dataframe
X_new = dfle[['town','area']].values
print(X_new)

y_new= dfle.price

ohe = OneHotEncoder(categorical_features=[0])

X_new=ohe.fit_transform(X_new).toarray()
print(X_new)