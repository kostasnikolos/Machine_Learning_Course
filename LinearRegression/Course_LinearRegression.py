# Uesed to solve problems like predicting home prices, predicting weather, predicting stock price (PREDICTED VALUE IS CONTINUOUS)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
print(sklearn.__version__)

# import data frame
df = pd.read_excel(r'C:\Users\kosta\OneDrive - Πολυτεχνείο Κρήτης\Μαθήματα\10ο Εξάμηνο\Ml_Python_Course\Book1.xlsx')
# print(df.head())

# Display the column names of the DataFrame
# print(df.columns)
# perforom linear regression
reg = linear_model.LinearRegression()
# train reg
reg.fit(df[['area']],df.price)

#  print the date on a plot
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price, color='red',marker='+')
# print prices areas and also the predicted value of prices for each area value
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()


pred=reg.predict([[3300]])
# print(pred)

# print coef(a) y =ax + b
# print(reg.coef_)

# print b
# print(reg.intercept_)

d = pd.read_excel(r'C:\Users\kosta\OneDrive - Πολυτεχνείο Κρήτης\Μαθήματα\10ο Εξάμηνο\Ml_Python_Course\areas.xlsx')
d.head(3)
# predict the values for the new areas of the new file 
p=reg.predict(d)
print(p)

# sav the predictions by creating a new column on the data array
d['prices']=p
# print(d)

# import the new predictions on a xl file 
d.to_excel("prediction.xlsx")

# ################################## EXERCISE###################
# Goal is to predict the canadian price per person on 2020


# import the data first
dff= pd.read_csv(r'C:\Users\kosta\OneDrive - Πολυτεχνείο Κρήτης\Μαθήματα\10ο Εξάμηνο\Ml_Python_Course\canada_per_capita_income.csv')
# Get the column names
# print(dff.head(4))
# print(dff.columns)
reg.fit(dff[['Year']],dff.Income)


#  print the date on a plot
plt.xlabel('Year')
plt.ylabel('Income')
plt.scatter(dff.Year,dff.Income, color='red',marker='+')
# print incomess years and also the predicted income  for each year 
plt.plot(dff.Year,reg.predict(dff[['Year']]),color='blue')
plt.show()

