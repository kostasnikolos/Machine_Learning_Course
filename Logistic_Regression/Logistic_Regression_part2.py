import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits= load_digits()


# every data in the data set is aan array

# print the image
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

# print(digits.target[0:5])

# split the data
X= digits.data
y=digits.target
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2)

#  print the length of the splitted datasetss
# print(len(X_train))
# print(len(X_test))

# lets train our model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)

#  show the accuracy of the model
# print(model.score(X_test,y_test))


#  now predict a random int USING DATA NOT IMAFGE
import random

random_number = random.randint(0, 9)
# print("target is:",digits.target[random_number])
# print("model gave:",model.predict([X[random_number]]))


# ############## CONFUSION MATRIX ######################
#  letcs check all the accuracy of the model using CONFUSION MATRIX
y_predicted= model.predict(X_test)
# get confusion matrix
from sklearn.metrics import confusion_matrix

#  in the confusion matrix rows represent the acutal vales and columns the predicted ones. 
# FOr example if at raw 9 column 2 we see 10 it means that 13 times the model predicted its a 2 when it was actually a 9
#  Diagonal elements should be hight which mean they are correct and the other should be 0
conf_matrix = confusion_matrix(y_test,y_predicted)
# print(conf_matrix)

import seaborn as sn
plt.figure(figsize= (10,7))
sn.heatmap(conf_matrix,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True Values')
plt.show()