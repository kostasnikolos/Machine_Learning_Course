# PROBLEM
#  We have to predict home prices again but after we have build the line of the data
#  we have to find  that line . In a more complicated problem there will be more lines than ont|

# SOLUTION with MEAN SQUARE ERROR (Cost Function)
#  Draw any random data line and then calculate the error from the actual point to your date point (DELATA) and square it (because the differencs might be negative).
# Sum all up and then devide by n  forming the mean square error


# Gradient Descent
# Way to find the best starting line y=mx = b
# maybe start with m=0 b=0  calculate the cost and the reduce the values of m,b  at every step untill you reach the minimun error and get the correct m,b

#  A way to do gradient decent is to start with a big step and make it smaller at every step(learning rate) by calculating the slope 
#  we calculate the partial derivative of m and b for the mean square error

import numpy as np
import matplotlib.pyplot as plt
import math

def gradient_decent(x,y):
    # start with m,b=0
    m_current = b_current = 0
    n = len(x)
    iterations_num = 10000
    # define learning rate 
    learning_rate= 0.08
    previous_cost= 0
    
    for i in range(iterations_num):
        #  y =mx+ b
        y_predicted = m_current * x + b_current
        

        # calculate cost of the predicted function
        cost= (1/n)*sum([val**2 for val in (y-y_predicted)])
        
        #  Now we want to find the perfect number of iterations to bring the corrrect answer
        # check if the number of iterations is already enough
        if math.isclose(previous_cost ,cost, rel_tol=1e-20, abs_tol=0.0):
            print(" Its enough now break")
            print("iterations num is {}".format(i))
            break
        
        #  calculate  derivatives
        md= -(2/n)*sum(x*(y-y_predicted))
        bd= -(2/n)*sum(y-y_predicted)
        
        #  move to the next tep
        m_current= m_current - learning_rate* md
        b_current = b_current - learning_rate*bd
        
        # update the previous cost
        previous_cost= cost
        
        
        #  print the values
        # print("current m {}  , current b {}, cost {} ,iteration {}".format(m_current,b_current,cost,iterations_num))
        #  print also the carrent cost
        
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])



gradient_decent(x,y)

#  tolerannce 1e-20
