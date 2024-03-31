import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

flowers= load_iris()


X = flowers.data
y = flowers.target

# Create a scatterplot of pairs of features
plt.figure(figsize=(12, 6))

# Plot Sepal Length vs. Sepal Width
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Setosa', c='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Versicolor', c='green')
plt.scatter(X[y == 2, 0], X[y == 2, 1], label='Virginica', c='blue')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs. Sepal Width')
plt.legend()
plt.show()