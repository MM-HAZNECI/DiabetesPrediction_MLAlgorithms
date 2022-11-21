#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np

#Loading Dataset
df=pd.read_csv("diabetes.csv")
df.head()
df.info()

df.isnull().sum()

df.describe().T


#How many zeros does each category contain?
df.eq(0).sum()

#We do not change anything about pregnancies and outcome because inputs might be 0 in "Pregnancies" and "Outcome" columns.


#Looking for colmuns which includes 0 
df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]



df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]=df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]].replace(0,np.NaN)



df.eq(0).sum()


#Filling NaN blocks with mean value
df.fillna(df.mean(),inplace=True)


df.corr()


import seaborn as sns
sns.heatmap(df.corr())





#Target variable is "Outcome" and other variables are feauture variables.
x_cols = ['Glucose','Insulin','BMI','Outcome']
df=df[x_cols]
df
x=df.drop(["Outcome",],axis=1)
x

y=df.Outcome
y


# Splitting Data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# ## Decision Tree Model
model=DecisionTreeClassifier()
#Train Classifier
model=model.fit(x_train,y_train)
#Predict 
y_pred=model.predict(x_test)


#Visualization DecisionTree Model
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


#Evaluation using Accuracy Score 
from sklearn import metrics 
dt_accuracy=metrics.accuracy_score(y_test,y_pred)*100
print("Accuracy:",dt_accuracy)



#Checking prediction values
model.predict([[110,80,30.3]])
model.predict([[300,110,35.6]])


# ## KNN Algorithm

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



#Looking for knn n values
n= 100
output = knn(train_x,train_y,test_x,test_y,n)
n_range = range(1, n)
plt.plot(n_range, output)





#Looking for knn n values
n= 200
output = knn(train_x,train_y,test_x,test_y,n)
n_range = range(1, n)
plt.plot(n_range, output)





#Looking for knn n values
n=300
output = knn(train_x,train_y,test_x,test_y,n)
n_range = range(1, n)
plt.plot(n_range, output)





#Evaluating Model
knn_acc=np.max(output)*100
print("Accuracy is: ",knn_acc)




#Checking prediction values
model.predict([[110,80,30.3]])




#Checking prediction values
model.predict([[300,110,35.6]])


# ## Random Forest Algorithm




#Splitting Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=1)



#Genarating and Evaluating Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc.predict(x_test)
rfc_acc=rfc.score(x_test, y_test)*100
print("Accuracy is: ",rfc_acc)



#Checking prediction values
model.predict([[110,80,30.3]])




#Checking prediction values
model.predict([[300,110,35.6]])


# ## Supported Vector Machine Algorithm




#Splitting Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.2,random_state=1)


#Genarating and Evaluating Model
from sklearn import svm
supvm=svm.SVC(kernel="linear")
supvm.fit(x_train,y_train)
y_pred=supvm.predict(x_test)
#Evaluating
from sklearn import metrics
supvm_acc=metrics.accuracy_score(y_test, y_pred)*100
print("Accuracy is: ",supvm_acc)


#Checking prediction values
model.predict([[110,80,30.3]])

#Checking prediction values
model.predict([[300,110,35.6]])


# ### According to accuracy rate finding best algorithm


daata=[dt_accuracy,knn_acc,rfc_acc,supvm_acc]
df2 = pd.DataFrame(daata, columns=['Accuracy'])
df2.index = ['Decision Tree', 'KNN', 'Random Forest', 'Support Vector Machine']
df2


#Finding max accuracy value
maxvalue = df2.idxmax(0)
print("The most suitable algorithm is: ",maxvalue)






