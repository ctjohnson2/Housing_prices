#!/usr/bin/python3

import defs
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)


def plot_learning(model,x,y):

  x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2)
  train_err, val_err,sam_size = [],[],[]
  for i in range(1,len(x_train),100):
    sam_size.append(i)
    model.fit(x_train[:i],y_train[:i])
    y_train_predict = model.predict(x_train[:i])
    y_val_predict = model.predict(x_val)
    train_err.append(mean_squared_error(y_train[:i],y_train_predict[:i]))
    val_err.append(mean_squared_error(y_val,y_val_predict))
  plt.axis([0,1200,0,100000])
  plt.plot(sam_size,np.sqrt(train_err),"r-+",linewidth=2,label="train")
  plt.plot(sam_size,np.sqrt(val_err),"b-",linewidth=3,label="val")
  plt.show()

def main():

#  housing_train = defs.prep_sets("test.csv","train.csv")
#  attributes = []
#  for col in housing_train.columns:
#    attributes.append(col)

#  sale_min = housing_train["SalePrice"].min()
#  sale_max = housing_train["SalePrice"].max()


#  price_train = housing_train["SalePrice"]
#  housing_train = housing_train.drop("SalePrice",axis=1)

#  sale_mean = price_train.mean()
#  price_train = price_train/price_train.mean()
#  housing_train, housing_val, price_train, price_val = train_test_split(housing_train,price_train, test_size = 0.2)
  housing_train, housing_val, price_train, price_val, sale_mean = defs.prep_strat_set("train.csv")

  # grid search

  from sklearn.model_selection import GridSearchCV
  forest_reg = RandomForestRegressor()
  param_grid = [{'n_estimators': [3,10,30,50,70], 'max_features':
  [2,4,6,8,10,12,14,16,18,20]},{'bootstrap':[False], 'n_estimators':[3,10],'max_features':
  [2,3,4]},]

  grid_search = GridSearchCV(forest_reg, param_grid, cv=2,scoring='neg_mean_squared_error',return_train_score=True)
  grid_search.fit(housing_train,price_train)
  print(grid_search.best_params_)

  forest_reg = RandomForestRegressor(bootstrap=True,max_features=20,n_estimators=70)
  forest_reg.fit(housing_train,price_train)
  forest_rmse = sale_mean*np.sqrt(mean_squared_error(price_train,forest_reg.predict(housing_train)))
  scores = cross_val_score(forest_reg, housing_train,price_train,scoring="neg_mean_squared_error",cv=5)
  forest_rmse_scores = sale_mean*np.sqrt(-scores)
  defs.display_scores(forest_rmse_scores)
  
  forest_rmse_train = np.sqrt(mean_squared_error(forest_reg.predict(housing_train),price_train))
  forest_rmse_test = np.sqrt(mean_squared_error(forest_reg.predict(housing_val),price_val))

  print("FULL","test=",sale_mean*forest_rmse_test,"train=",sale_mean*forest_rmse_train)
if __name__ == "__main__":

  main()
