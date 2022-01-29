#!/usr/bin/env python
# coding: utf-8

# In[1]:


#predicting stock prices using simple ML model
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[2]:


#get the stock data for facebook
df = quandl.get("WIKI/FB")


# In[4]:


df.tail()


# In[13]:


#get the adjust close price
df = df[['Adj. Close']]


# In[14]:


df.head()


# In[20]:


#forescasting how many days out in the future
forecast_out = 30

#create another column (the target or dependent variable) shifted 'n' units up, the shift() functions shift data by 30 day so -30
df['Predictions'] = df.shift(-forecast_out)


# In[21]:


#moved up the data in predictions by 1 day of the next day adj. Close
df.tail()


# In[23]:


#create the independent data set (X)
#convert the dataframe to a numpy array
X = np.array(df.drop(['Predictions'],1))
#remove the last 'forecast_out' rows
X = X[:-forecast_out]


# In[24]:


X


# In[25]:


#Create the dependent data set (y)
#convert the dataframe to a numpy array (All of the values including the Nan's)
y = np.array(df['Predictions'])

#get all of the y values except the last 'n' rows
y = y[:-forecast_out]


# In[26]:


y


# In[27]:


#Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[28]:


#Create and train the support vector machine (regressor)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

#train the model

svr_rbf.fit(x_train, y_train)


# In[30]:


#testing model: Score returns the coefficient of determination R^2 of the prediction
#the best possible score is 1.0

svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)


# In[31]:


#create and train the linear regression model

lr = LinearRegression()

#train the model
lr.fit(x_train, y_train)


# In[32]:


#testing model: Score returns the coefficient of determination R^2 of the prediction
#the best possible score is 1.0

lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)


# In[33]:


#set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Predictions'],1))[-forecast_out:]


# In[34]:


x_forecast


# In[35]:


#print the linear regression model predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)


# In[36]:


#print the support vector regressor model predictions for the next 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)


# In[ ]:




