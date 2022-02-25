
#Long Short Term Memory (Aritificial Recurrent Neural Network) to predict closing stock price of a corporation

import math
import pandas_datareader as web
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#Get stock quote
yf.pdr_override()

start = datetime.strptime('2013-01-01', '%Y-%m-%d')
end = datetime.strptime('2021-12-17', '%Y-%m-%d')
df = pdr.get_data_yahoo("AMZN", start, end)


#get the number of rows and columns in the data set
df.shape


# In[182]:


#visualize the closing price history
plt.figure(figsize = (16,8))
plt.title('Historical Closing Price')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.show()


# In[183]:


#create new dataframe with only the 'close' column
data = df.filter(['Close'])
#convert the dataframe to a numpy array
dataset = data.values
#get the number of rows to train the model with (80% of the dataset)
training_data_len = math.ceil(len(dataset) * .8)

training_data_len


# In[184]:


#Scale the Data
scaler = MinMaxScaler(feature_range = (0,1))
#range is 0,1 inclusive and transforms the dataset within these two values
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[185]:


#Create the training data set
#Create the scaled training data set and get back all the columns
#the array part is the x_train and the second output is the y_train dataset
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()


# In[186]:


#convert the x_train and y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)


# In[187]:


#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[188]:


#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[189]:


#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[190]:


#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[191]:


#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60: , :]

#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# In[192]:


#Convert the data to a numpy array
x_test = np.array(x_test)


# In[193]:


#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[194]:


#Get the models predicted price values with inverse transfrom value of x and y
#Want predictions to contains the same values of our y_test dataset
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[195]:


#Create new model, get the root mean squared error (RMSE) 
#if rmse = 0 then the model prediction is exact, the closer it is to 0 the more accurate
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[196]:


#Plot the data with validation dataset
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Prediction Model')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
plt.show


# In[197]:


#Show the valid and predicted prices 
valid


# In[199]:


#Get the quote
start = datetime.strptime('2013-01-01', '%Y-%m-%d')
end = datetime.strptime('2021-12-17', '%Y-%m-%d')

apple_quote = pdr.get_data_yahoo("AAPL", start, end)

#Create a new data_frame
new_df = apple_quote.filter(['Close'])
#Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create empty list
z_test = []
#Append the past 60 days
z_test.append(last_60_days_scaled)
#Convert the z_test data set to a numpy array
z_test = np.array(z_test)
#Reshape the data
z_test = np.reshape(z_test, (z_test.shape[0], z_test.shape[1], 1))
#Get the predicted scale price
pred_price = model.predict(z_test)
#Undo scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[ ]:




