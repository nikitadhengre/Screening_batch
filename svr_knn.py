import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#---------------------------------
dataset=pd.read_csv(r"C:\Users\hp\Downloads\emp_sal.csv")

x =dataset.iloc[:,1:2].values

y =dataset.iloc[:,2].values
#---------------------------------
#fiiting svr model
from sklearn.svm import SVR

regressor=SVR(kernel='poly',degree=4,gamma='auto')
regressor.fit(x,y)

y_pred_svr=regressor.predict([[6.5]])
print(y_pred_svr)

#--------------------------------------------
#fit knn modekl to my dataset
from sklearn.neighbors import KNeighborsRegressor
reg_knn=KNeighborsRegressor(n_neighbors=3,p=3,leaf_size=30)
reg_knn.fit(x,y)


y_pred_knn=reg_knn.predict([[6.5]])
y_pred_knn

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#if you check the output that is svr model & its predicting the each of the real observation
#red points are real observation point & blue lines are predicted line & now you can say svr is fitted much better curve on the dataset
#same hear if you check the ceo actual observation point but you will find as still we can improve the graph and lets see how can we do that in svr
#in this case ceo is outlier hear becuase ceo is quite far from our observation, thats ok

#what exactly we are doing hear to check the what exactly employees have 6.5yrs experience predict salary


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#------------------------------------
from sklearn.tree import DecisionTreeRegressor
reg_dtr=DecisionTreeRegressor(criterion='poison',)
reg_dtr.fit(x,y)