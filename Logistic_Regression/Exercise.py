import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


# lest first import the data
df_old=pd.read_csv(r'C:\Users\kosta\OneDrive - Πολυτεχνείο Κρήτης\Μαθήματα\10ο Εξάμηνο\Ml_Python_Course\Logistic_Regression\HR_comma_sep.csv')

# lets also plot the data

# plt.xlabel('Age')
# plt.ylabel('Insurance')
# plt.scatter(df.age,df.bought_insurance, color='red',marker='+')
# plt.show()

# S
left = df_old[df_old.left==1]
# print("people left",left.shape)

retained = df_old[df_old.left==0]
# print("people stayed",retained.shape)

#we need ro replace the salary with real numbers

# First, create a mapping dictionary for the salary categories
salary_mapping = {'low': 0, 'medium': 1, 'high': 2}

# Replace the values in the 'salary' column using the mapping dictionary
df_old['salary'] = df_old['salary'].replace(salary_mapping)

#  now drop the unused departmen
df= df_old.drop(columns=['Department'])


#  print the average of all categories for people who left
averages=df.groupby('left').mean()
# print (averages)


# we can see that monthly hours, time spend ,promotion and salary affects

#  plot salary impact
impact1=pd.crosstab(df.salary,df.left).plot(kind='bar')
# plt.show()

#  now lets train the model

# first keep the columns we are intrested
subdf= df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]


X= subdf

y=df.left


# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


# train
model = LogisticRegression()
model.fit(X_train, y_train)

model.predict(X_test)


score= model.score(X_test,y_test)
print(score)