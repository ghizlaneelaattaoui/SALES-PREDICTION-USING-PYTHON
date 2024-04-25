#!/usr/bin/env python
# coding: utf-8

# # Reading and Understanding the Data

# In[43]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings


# In[44]:


data=pd.read_csv(r"C:\Users\Hp\Desktop\projet-data science\codsoft data science\advertising.csv", encoding="latin-1")
data


# In[30]:


data.head()


# In[45]:


warnings.simplefilter(action='ignore', category=FutureWarning)
os.getcwd()


# # Data Inspection

# In[31]:


data.shape


# In[32]:


data.info()


# In[33]:


data.describe()


# # Data Cleaning

# In[34]:


data.isnull().sum()


# In[47]:


sns.pairplot(data, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales", kind="reg")


# # Exploratory Data Analysis

# In[48]:


data.hist(bins=20)


# In[ ]:





# In[38]:


sns.heatmap(data.corr(), cmap="YlGnBu", annot = True)
plt.show()


# # Model Building

# In[39]:


X = data['TV']
y = data['Sales']


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[41]:


X_train.head()


# In[42]:


y_train.head()


# In[51]:


# Linear Regression Model

lin_model = sm.ols(formula="Sales ~ TV + Radio + Newspaper", data=data).fit()


# In[52]:


print(lin_model.params, "\n")


# In[53]:


results = []
names = []


# In[55]:


models = [('LinearRegression', LinearRegression())]


# In[57]:


# Make predictions on new data

new_data = pd.DataFrame({'TV': [100], 'Radio': [50], 'Newspaper': [25]})
predicted_sales = lin_model.predict(new_data)
print("Predicted Sales:", predicted_sales)


# In[ ]:




