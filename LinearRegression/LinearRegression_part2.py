import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn import linear_model
from word2number import w2n
# import data frame
#  linear euation should look like price= m1*area + m2*bedrooms + m3*age + b
df = pd.read_csv(r'C:\Users\kosta\OneDrive - Πολυτεχνείο Κρήτης\Μαθήματα\10ο Εξάμηνο\Ml_Python_Course\homeprices.csv')

# fint the median of bedrooms
median_bedrooms= math.floor(df.bedrooms.median())
median_bedrooms= int(median_bedrooms)
# we first need to handle the missing data point
# we will fill it with the median
df['bedrooms'] = df['bedrooms'].fillna(median_bedrooms)

reg=linear_model.LinearRegression()
# train , use [[independent variables]], df.dependent
reg.fit(df[['area','bedrooms','age']],df.price)
# print(reg.coef_)
# print(reg.intercept_)
pr=reg.predict([[3000,3,15]])
# print(pr)

# import data frame
#  linear euation should look like price= m1*area + m2*bedrooms + m3*age + b
dff = pd.read_csv(r'C:\Users\kosta\OneDrive - Πολυτεχνείο Κρήτης\Μαθήματα\10ο Εξάμηνο\Ml_Python_Course\hiring.csv')
print(dff.columns)
print(dff)
#  we first need to transforrm text on the first column to numbers
# Replace text representations of numbers with their numerical equivalents in the 'experience' column
dff['experience'] = dff['experience'].apply(lambda x: w2n.word_to_num(x) if isinstance(x, str) else x)



# we will fill the 0 experiences with 0 
dff['experience'] = dff['experience'].fillna(0)


# now we need to calculate the median for the scores 
# fint the median of bedrooms
median_scores= math.floor(dff.scores.median())
median_scores= int(median_scores)
# we first need to handle the missing data point
# we will fill it with the median
dff['scores'] = dff['scores'].fillna(median_scores)

# Print the DataFrame after the replacement
print(dff)

#  now lets train
reg.fit(dff[['experience','scores','interview_score']],dff.salary)


pr=reg.predict([[2,9,6]])
print(pr)
