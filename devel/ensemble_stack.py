#!/usr/bin/python3

import defs
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)

def plot_result(pred,y):

  plt.axis([0,50,0,1000000])
  plt.plot(pred,"ob",label="Prediction")
  plt.plot(y,"or",label="Price")
  plt.show()

def print_rmse(model,train,price_train,val,price_val,mean_):

  vote_rmse_train = mean_*np.sqrt(mean_squared_error(price_train,model.predict(train)))
  vote_rmse_val = mean_*np.sqrt(mean_squared_error(price_val,model.predict(val)))
  print("Ensemble:",vote_rmse_train,vote_rmse_val)
def main():

  # voting bag method with, lin_reg, ran_for, svm_poly, svm_rbf
  
  housing_data = defs.prep_sets("test.csv","train.csv")
  attributes = []
  for col in housing_data.columns:
    attributes.append(col)

  sale_min = housing_data["SalePrice"].min()
  sale_max = housing_data["SalePrice"].max()

  housing_prices = housing_data["SalePrice"]
  housing_data = housing_data.drop("SalePrice",axis=1)

  sale_mean = housing_prices.mean()
  housing_prices = housing_prices/housing_prices.mean()
  housing_train, housing_val, price_train, price_val = train_test_split(housing_data,housing_prices, test_size = 0.2)

  from sklearn.model_selection import StratifiedShuffleSplit
  #split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=1)
  #for train_ind, test_ind in split.split(housing_data,housing_values):

   # housing_train = housing_data.loc[train_ind]
   # price_train = housing_values.loc[train_ind]
   # housing_val = housing_data.loc[test_ind]
   # price_val = housing_vales.loc[test_ind]

  test_dat, train_dat = defs.read_sets("test.csv","train.csv")

  train_dat = defs.clean_up_nans(train_dat)
  from sklearn.model_selection import StratifiedShuffleSplit
  split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)

  for train_ind, test_ind in split.split(train_dat,train_dat["Neighborhood"]):
    train_list = train_ind
    test_list = test_ind

  housing_data = defs.encode_data(train_dat)
  housing_prices = housing_data["SalePrice"]
  housing_data = housing_data.drop("SalePrice",axis=1)

  sale_mean = housing_prices.mean()
  housing_prices = housing_prices/housing_prices.mean()

  housing_train, price_train = housing_data.iloc[train_list],housing_prices.iloc[train_list]
  housing_val, price_val = housing_data.iloc[test_list], housing_prices.iloc[test_list]

  from sklearn.linear_model import Ridge
  ridge_reg = Ridge(alpha=1,solver="cholesky")
  lin_reg = LinearRegression()
  svm_poly_reg = SVR(kernel="poly", degree=4, C=1e3)
  svm_rbf_reg = SVR(kernel="rbf", C=1e6, epsilon=0.01) 
  svm_rbf_reg = svm_poly_reg
  forest_reg = RandomForestRegressor(bootstrap=True,max_features=20,n_estimators=70)
  from sklearn.ensemble import GradientBoostingRegressor

  gbr = GradientBoostingRegressor()

  subset1_data, subset2_data, subset1_values, subset2_values = train_test_split(housing_train,price_train, test_size = 0.1)

#  lin_reg.fit(subset1_data,subset1_values)
  ridge_reg.fit(subset1_data,subset1_values)
  forest_reg.fit(subset1_data,subset1_values)
  gbr.fit(subset1_data,subset1_values)  
  
  blender_input = np.stack((ridge_reg.predict(subset2_data),forest_reg.predict(subset2_data),gbr.predict(subset2_data)),axis=-1)

  #blender = SVR(kernel="poly", degree=4, C=1e3)
  blender = RandomForestRegressor()
  blender.fit(blender_input,subset2_values)

  val_input = np.stack((ridge_reg.predict(housing_val),forest_reg.predict(housing_val),gbr.predict(housing_val)),axis=-1)
  blender_rmse = np.sqrt(mean_squared_error(blender.predict(val_input),price_val))

  print("RF:",blender_rmse*sale_mean)

  import xgboost

  xgb = xgboost.XGBRegressor()

  blender = xgb
  blender.fit(blender_input,subset2_values)

  
  blender_rmse_norm = np.sqrt(mean_squared_error(blender.predict(val_input),price_val))
  
  blender_pred = blender.predict(val_input)*sale_mean
  price_val_true = price_val*sale_mean

  blender_rmse = np.sqrt(mean_squared_error(blender_pred,price_val_true))

  print("XGB:",blender_rmse,blender_rmse_norm)
  

  
  print(np.shape(blender_pred),np.shape(price_val_true.values))
  #plot_result(blender_pred,price_val_true.values)

  
  blender = SVR(kernel="poly", degree=2, C=1)
  blender.fit(blender_input,subset2_values)

  
  blender_rmse_norm = np.sqrt(mean_squared_error(blender.predict(val_input),price_val))

  blender_pred = blender.predict(val_input)*sale_mean
  price_val_true = price_val*sale_mean

  blender_rmse = np.sqrt(mean_squared_error(blender_pred,price_val_true))

  print("SVM:",blender_rmse,blender_rmse_norm)



  print(np.shape(blender_pred),np.shape(price_val_true.values))
  plot_result(blender_pred,price_val_true.values)
if __name__=="__main__":

  main()
