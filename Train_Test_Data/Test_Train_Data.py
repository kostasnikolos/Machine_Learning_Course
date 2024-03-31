#  we need the 80% percent of the datta for training and the remaining 20% data for testing

#  so we will split the data into two parts

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# in order to split
from sklearn.model_selection import train_test_split

# import data
df = pd.read_csv("C:\\Users\\kosta\\OneDrive - Πολυτεχνείο Κρήτης\\Μαθήματα\\10ο Εξάμηνο\\Ml_Python_Course\\Train_Test_Data\\carprices.csv")


# print(df)
plt.xlabel('Mileage')
plt.ylabel('Sell Price')
plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
# plt.show()

X= df[['Mileage','Age(yrs)']]
X= df[['Mileage','Age(yrs)']]
X= df[['Mileage','Age(yrs)']]
X= df[['Mileage','Age(yrs)']]
y= df['Sell Price($)']

print("HELLOWORLD")
# Splitting the data into training and testing sets (80% train, 20% test)
# using random state means the samples will not change in every exevution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# train the model
clf= LinearRegression()
clf.fit(X_train,y_train)

# predict
# petros

pr=clf.predict(X_test)

#  test the result using score method
score=clf.score(X_test,y_test)
print(score)