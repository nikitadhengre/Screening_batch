import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


dataset=pd.read_csv(r"C:\Users\hp\Salary_Data.csv")
'''
x =dataset.iloc[:,:-1].values

y =dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,test_size=0.20,random_state=0)



from sklearn.linear_model import LinearRegression

regressor= LinearRegression()

regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)



plt.scatter(x_test, y_test, color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("salary vs experience(Training set)")
plt.xlabel("year of experience")
plt.ylabel("salary")
plt.show()



m_slope=regressor.coef_
m_slope


c_intercept=regressor.intercept_
c_intercept



emp_15_predict_salary=m_slope*15+c_intercept
emp_15_predict_salary

'''














