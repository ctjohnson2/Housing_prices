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
from warnings import simplefilter
import matplotlib.pyplot as plt

simplefilter(action='ignore',category=FutureWarning)

def print_rmse(model,train,price_train,val,price_val,mean_):
#  mean_=1
  vote_rmse_train = mean_*np.sqrt(mean_squared_error(price_train,model.predict(train)))
  vote_rmse_val = mean_*np.sqrt(mean_squared_error(price_val,model.predict(val)))
  print(str(model),":",vote_rmse_train,vote_rmse_val)
def main():
 
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
  
  test_dat, train_dat = defs.read_sets("test.csv","train.csv")

  train_dat = defs.clean_up_nans(train_dat)
  from sklearn.model_selection import StratifiedShuffleSplit
  split = StratifiedShuffleSplit(n_splits=1,test_size=0.1, random_state=42)

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

  lin_reg = LinearRegression()
  from sklearn.linear_model import Ridge
  ridge_reg = Ridge(alpha=1,solver="cholesky")
  svm_poly_reg = SVR(kernel="poly", degree=4, C=1e3)
  svm_rbf_reg = SVR(kernel="rbf", degree=12, C=1e6, epsilon=0.01) 
  forest_reg = RandomForestRegressor(bootstrap=True,max_features=20,n_estimators=70)
  from sklearn.ensemble import GradientBoostingRegressor

  gbr = GradientBoostingRegressor(max_depth=3,n_iter_no_change=5,validation_fraction=0.05)
  import xgboost
  xgb = xgboost.XGBRegressor()
  
  ridge_reg.fit(housing_train,price_train)
  gbr.fit(housing_train,price_train)
  xgb.fit(housing_train,price_train)
  forest_reg.fit(housing_train,price_train)
  vote = VotingRegressor(estimators=[('lr',ridge_reg),('gbr',gbr),('fr',forest_reg)])
  print(housing_data.info())
  print("Fitting model")
  
  vote.fit(housing_train,price_train)
  
  print_rmse(ridge_reg,housing_train,price_train,housing_val,price_val,sale_mean)
  print_rmse(gbr,housing_train,price_train,housing_val,price_val,sale_mean)
  print_rmse(forest_reg,housing_train,price_train,housing_val,price_val,sale_mean)
  print_rmse(vote,housing_train,price_train,housing_val,price_val,sale_mean) 
  
  housing_val_pred = vote.predict(housing_val)
  lin_mse_val =  np.sqrt(mean_squared_error(housing_val_pred,price_val))*sale_mean

  x_ = range(len(housing_val_pred))
  plt.bar(x_,sale_mean*housing_val_pred,color='red',alpha=0.5,label='prediction')
  plt.bar(x_,sale_mean*price_val,color='blue',alpha=0.5,label='actual')
  plt.ylabel("Prices in $")
  plt.title("Validation Results")
  plt.text(0,4e5,"RMSE= "+str(lin_mse_val))
  plt.legend()
  plt.show()


  '''g_type = housing_val["GarageType"]*6

  

  for j in range(1,7):
    one = []
    pred = []
    for i in range(len(g_type)):
    
      if g_type.values[i]==j:
       
       one.append(price_val.iloc[i])
       pred.append(vote.predict(housing_val)[i])
 
 
    rmse = np.sqrt(mean_squared_error(one,pred))
    print("Garage",j,rmse*sale_mean,len(one))


  g_type = housing_val["SaleCondition"]*5
  
  
  for j in range(1,6):
    one = []
    pred = []
    for i in range(len(g_type)):
   
      if g_type.values[i]==j:

       one.append(price_val.iloc[i])
       pred.append(vote.predict(housing_val)[i])

    if len(one)!=0:

         rmse = np.sqrt(mean_squared_error(one,pred))
         print("SaleCond",j,rmse*sale_mean,len(one))
'''
if __name__=="__main__":

  main()
