#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ticker = 'PG'
data = pd.DataFrame()
data[ticker] = wb.DataReader(ticker, data_source='yahoo', start = '2007-1-1')['Adj Close']


# In[3]:


#pandas.pct_change() obtains simple returns from a provided dataset
log_returns = np.log(1 + data.pct_change())


# In[4]:


log_returns.tail()


# In[5]:


data.plot(figsize=(10,6));
#shows PG price


# In[6]:


log_returns.plot(figsize = (10,6))
#shows PG returns, show that it has stable means


# In[7]:


#the average log return
u = log_returns.mean()
u


# In[9]:


var = log_returns.var()
var


# In[10]:


#we know that drift = u - 1/2 X var
drift = u - (0.5 * var)
drift


# In[11]:


#stand deviation of log return
stdev = log_returns.std()
stdev


# In[12]:


#Brownian motion -> r = drift + stdev * e^r


# In[13]:


type(drift)


# In[14]:


type(stdev)


# In[16]:


#put drift into numpy array, object.value does the same
np.array(drift)


# In[17]:


#object.values transfers the object into a numpy array
drift.values


# In[18]:


stdev.values


# In[19]:


#Z in the brownian motion corresponds to the distance between the mean and the events, expressed as the number of standard deviations
norm.ppf(0.95)


# In[20]:


x = np.random.rand(10, 2)
x


# In[21]:


#to obtain the distance from the mean corresponding to the above random generated possibilities
norm.ppf(x)


# In[22]:


z = norm.ppf(np.random.rand(10, 2))
z


# In[23]:


#time interval is 1000, forecasting upcoming 1000 days, iterations is 10 series of future stock price predictions
t_intervals = 1000
iterations = 10


# In[24]:


#daily_returns = e^r
#r = drift + stdev x z
#numpy.exp() -> calculates e^(expression)
daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals,iterations)))


# In[25]:


daily_returns
#generation 10 columns of random stocks prices -> 1000 rows (for each upcoming day)


# In[26]:


#Creating a price list
#Price of date t = S(t) = previous day price times simulated daily return
#S(t) = S(0) x daily_returns(t)
#S(t+1) = S(t) x daily_returns(t+1)
#S(t+999) = S(t+999) x daily_returns(t+999)


# In[27]:


#iloc[-1] takes value of the last item on the table = the most current price data of PG
S0 = data.iloc[-1]
S0


# In[30]:


#price list can only be as big as the daily_returns matrix
#zero_likes fill everything in list of size (daily_returns) with zeroes
price_list = np.zeros_like(daily_returns)


# In[31]:


price_list


# In[32]:


price_list[0] = S0
price_list
#this makes the first array all filled with the most current PG price -> all 1000 rows of the first 10 possibilities


# In[35]:


#S(t) = S(t-1) x daily_returns(t)
#iteration this 1000 times
for t in range(1, t_intervals):
    price_list[t] = price_list[t-1] * daily_returns[t]


# In[36]:


price_list


# In[37]:


plt.figure(figsize=(10,6))
plt.plot(price_list);
#shows the 10 possible future stocks prices for the next 1000 days


# In[ ]:




