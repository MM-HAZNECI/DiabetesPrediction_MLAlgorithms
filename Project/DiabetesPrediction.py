#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


#Loading Dataset
df=pd.read_csv("diabetes.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe().T


# In[6]:


#How many zeros does each category contain?
df.eq(0).sum()


# In[7]:


#We do not change anything about pregnancies and outcome because inputs might be 0 in "Pregnancies" and "Outcome" columns.


# In[8]:


#Looking for colmuns which includes 0 
df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]


# In[9]:


df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]=df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]].replace(0,np.NaN)


# In[10]:


df.eq(0).sum()


# In[11]:


#Filling NaN blocks with mean value
df.fillna(df.mean(),inplace=True)


# In[12]:


df.corr()


# In[13]:


import seaborn as sns


# In[14]:


sns.heatmap(df.corr())


# In[15]:


#Target variable is "Outcome" and other variables are feauture variables.


# In[16]:


x_cols = ['Glucose','Insulin','BMI','Outcome']
df=df[x_cols]
df
x=df.drop(["Outcome",],axis=1)
x


# In[17]:


y=df.Outcome
y


# Splitting Data

# In[18]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# ## Decision Tree Model

# In[19]:


model=DecisionTreeClassifier()
#Train Classifier
model=model.fit(x_train,y_train)
#Predict 
y_pred=model.predict(x_test)


# In[26]:


from six import StringIO
from IPython.display import Image
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
clf = DecisionTreeClassifier(random_state=1234,max_depth=4)
model = clf.fit(x, y)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=x_cols,  
                   class_names=["0","1"],
                   filled=True)


# In[21]:


#Evaluation using Accuracy Score 
from sklearn import metrics 
dt_accuracy=metrics.accuracy_score(y_test,y_pred)*100
print("Accuracy:",dt_accuracy)


# In[22]:


#Checking prediction values
model.predict([[110,80,30.3]])


# In[23]:


model.predict([[300,110,35.6]])


# ## KNN Algorithm

# In[37]:


import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
get_ipython().run_line_magic('matplotlib', 'inline')


#Splitting Data
split = 0.80 #80% train and 20% test dataset
total_len = len(df)
split_df = int(total_len*split)
train, test = df.iloc[:split_df,0:4],df.iloc[split_df:,0:4]
train_x = train[['Glucose','Insulin','BMI']]
train_y = train['Outcome']
test_x = test[['Glucose','Insulin','BMI']]
test_y = test['Outcome']


# In[38]:


def knn(x_train, y_train, x_test, y_test,n):
    n_range = range(1, n)
    results = []
    for n in n_range:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(x_train, y_train)
        predict_y = knn.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predict_y)
        results.append(accuracy)
    return results


# In[40]:


#Looking for knn n values
n= 100
output = knn(train_x,train_y,test_x,test_y,n)
n_range = range(1, n)
plt.plot(n_range, output)


# In[41]:


#Looking for knn n values
n= 200
output = knn(train_x,train_y,test_x,test_y,n)
n_range = range(1, n)
plt.plot(n_range, output)


# In[42]:


#Looking for knn n values
n=300
output = knn(train_x,train_y,test_x,test_y,n)
n_range = range(1, n)
plt.plot(n_range, output)


# In[57]:


#Evaluating Model
knn_acc=np.max(output)*100
print("Accuracy is: ",knn_acc)


# In[58]:


#Checking prediction values
model.predict([[110,80,30.3]])


# In[59]:


#Checking prediction values
model.predict([[300,110,35.6]])


# ## Random Forest Algorithm

# In[46]:


#Splitting Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=1)


# In[47]:


#Genarating and Evaluating Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc.predict(x_test)
rfc_acc=rfc.score(x_test, y_test)*100
print("Accuracy is: ",rfc_acc)


# In[48]:


#Checking prediction values
model.predict([[110,80,30.3]])


# In[49]:


#Checking prediction values
model.predict([[300,110,35.6]])


# ## Supported Vector Machine Algorithm

# In[50]:


#Splitting Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.2,random_state=1)


# In[51]:


#Genarating and Evaluating Model
from sklearn import svm
supvm=svm.SVC(kernel="linear")
supvm.fit(x_train,y_train)
y_pred=supvm.predict(x_test)
#Evaluating
from sklearn import metrics
supvm_acc=metrics.accuracy_score(y_test, y_pred)*100
print("Accuracy is: ",supvm_acc)


# In[52]:


#Checking prediction values
model.predict([[110,80,30.3]])


# In[53]:


#Checking prediction values
model.predict([[300,110,35.6]])


# ### According to accuracy rate finding best algorithm

# In[54]:


daata=[dt_accuracy,knn_acc,rfc_acc,supvm_acc]
df2 = pd.DataFrame(daata, columns=['Accuracy'])
df2.index = ['Decision Tree', 'KNN', 'Random Forest', 'Support Vector Machine']
df2


# In[55]:


#Finding max accuracy value
maxvalue = df2.idxmax(0)
print("The most suitable algorithm is: ",maxvalue)


# In[ ]:




