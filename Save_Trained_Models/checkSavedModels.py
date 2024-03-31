
import pickle
import joblib

# Check pickle
with open("model_pickle",'rb') as f:
    mp = pickle.load(f)
    
check=mp.predict([[3300]])
    
# Check joblib
mj = joblib.load('model_joblib')
check2=mj.predict([[3300]])
print("check joblib {} , check picckle {}".format(check2,check))