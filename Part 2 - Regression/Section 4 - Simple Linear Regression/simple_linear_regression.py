# importing the library 
import  numpy as  np
import  matplotlib.pyplot as plt
import pandas as  pd


#importing  the  data 

dataSet=pd.read_csv('Salary_Data.csv')
X=dataSet.iloc[:,:-1].values
y=dataSet.iloc[:,1].values


#splitting  the  dataSet into  training  set  and  test set

from  sklearn.model_selection import  train_test_split
X_train ,X_test, y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0,shuffle=True)


#fitting simple linear regression  to the Training set

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(X_train,y_train)

#prdicting the  test  set Results

y_pred=regressor.predict(X_test)

#Visualising the  Training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary  Vs Experiencce(Training  Set)")
plt.xlabel("Years  Of Experience")
plt.ylabel("Salary")
plt.show()

#Visualising the  Tesing  set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary  Vs Experiencce(Testing  Set)")
plt.xlabel("Years  Of Experience")
plt.ylabel("Salary")
plt.show()