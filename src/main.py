import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns



df=pd.read_csv("C:\MLProjects\LSTMmul\data\data.csv")

print("Features available in the data\n")
for i,name in enumerate(df.columns):
    print(str(i)+": "+str(name))
print("\n")


#Feature Engineering
mag=df["Speed"]
angle=df["WindDirection(Degrees)"]*(np.pi)/180
df["Speedx"]=mag*np.cos(angle)
df["Speedy"]=mag*np.sin(angle)

df["newtime"]=pd.to_datetime(df["UNIXTime"],unit='s')
df.sort_values(by="UNIXTime",inplace=True)

df["newindex"]=np.arange(df.shape[0])
df.set_index(keys="newindex",inplace=True)


day = 24*60*60
year = (365.2425)*day

df["daysin"]=np.sin(df["UNIXTime"]*2*np.pi/day)
df["daycos"]=np.cos(df["UNIXTime"]*2*np.pi/day)
df["yearsin"]=np.sin(df["UNIXTime"]*2*np.pi/year)
df["yearcos"]=np.cos(df["UNIXTime"]*2*np.pi/year)

#Select the columns to be used
df=df.filter(['daycos','daysin','Humidity',"Pressure","Radiation","Speedx","Speedy","Temperature"], axis=1)
print("Features used for the model\n")
for i,name in enumerate(df.columns):
    print(str(i)+": "+str(name))
print("\n")

print(df.tail())
#Scalling the data
scaler=StandardScaler()
df=scaler.fit_transform(df)

print("First element of transformed dataframe \n")
print(df[0])
print("\n")

#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 2. We will make timesteps = 3. 
#With this, the resultant n_samples is 5 (as the input data has 9 rows).
trainX = []
trainY = []

n_future = 2   # Number of samples we want to predict into the future
n_past = 14     # Number of past samples we want to use for predicting



for i in range(n_past, len(df) - n_future +1):
    trainX.append(df[i - n_past:i, 0:df.shape[1]])
    trainY.append(df[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
    










