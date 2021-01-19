import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns



df=pd.read_csv("C:\MLProjects\LSTMmul\data\data.csv")
print(df.columns)

for i,name in enumerate(df.columns):
    print(str(i)+": "+str(name))


#Feature Engineering
mag=df["Speed"]
angle=df["WindDirection(Degrees)"]*(np.pi)/180
df["Speedx"]=mag*np.cos(angle)
df["Speedy"]=mag*np.sin(angle)

df["newtime"]=pd.to_datetime(df["UNIXTime"],unit='s')
df.sort_values(by="UNIXTime",inplace=True)

df["newindex"]=np.arange(df.shape[0])
df.set_index(keys="newindex",inplace=True)







