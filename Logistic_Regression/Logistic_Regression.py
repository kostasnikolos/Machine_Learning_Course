#  Solve a simple CLASSIFICATION problem using Logistic Regression
#  Problems (CATEGORICAL) like spam , will customer buy or not, classifciation betweeen democratic repuplican independent.

# TYPES OF PROBLEMS OF CLASSIFICATION
#  customer will buy? Yes or No -> Binary Classification
#  a person is? Democratic or Repuplican or Independent? -> MultiClass Classification


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# lest first import the data
df=pd.read_csv(r'C:\Users\kosta\OneDrive - Πολυτεχνείο Κρήτης\Μαθήματα\10ο Εξάμηνο\Ml_Python_Course\Logistic_Regression\insurance_data.csv')
# print(df)

# lets also plot the data

plt.xlabel('Age')
plt.ylabel('Insurance')
plt.scatter(df.age,df.bought_insurance, color='red',marker='+')
# plt.show()



#  problem is we cant use linear regresion here since it may be problematic. We need binary classification using not a line but a SIGMOID OR LOGIT FUNCTION
#  sigmoid(z) = 1/ ( 1 + e^(-z)) ( comes up with a range between 0 to 1 as z goes up function goes to 1)
#  we can feed the y=mx +b to the sigmoid function giving:  y= 1/( 1 + e^-(mx +b))


#  we will first split our data using 

# x can be multivariable so we need double brackets
X= df[['age']]
y= df['bought_insurance']


# Splitting the data into training and testing sets (80% train, 20% test)
# using random state means the samples will not change in every exevution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.linear_model import LogisticRegression

# train the model
model = LogisticRegression()

model.fit(X_train,y_train)

pred=model.predict(X_test)

#  print score accuracy of the model
score= model.score(X_test,y_test)
# print("score ",score)

#  print probs of buying
preds=  model.predict_proba(X_test)
print(preds)