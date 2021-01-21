import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.preprocessing.sequence import TimeseriesGenerator



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

print("\nFirst element of transformed dataframe")
print(df[0])
print("\n")

# Train Validation and Test split
n=len(df)

train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]




#Input data must be in the shape n_samples x timesteps x n_features. 
trainX = []
trainY = []

n_future = 10   # Number of samples to the future
n_past = 100    # Number of samples from the past

for i in range(n_past, len(train_df) - n_future +1):
    trainX.append(train_df[i - n_past:i, 0:train_df.shape[1]])
    trainY.append(train_df[i:i + n_future, 4])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))



#model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()



# fit model
history = model.fit(trainX, trainY, epochs=2, batch_size=16, validation_split=0.1, verbose=1)


    
#Forecasting...
#Start with the last day in training date and predict future...
n_future=1500  #Redefining n_future to extend prediction dates beyond original n_future dates...

forecast = model.predict(trainX[-n_future:]) #forecast 
print(forecast)

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform





plt.plot(forecast)    
plt.show()             