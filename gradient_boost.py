#!/usr/bin/python3

import defs
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from warnings import simplefilter

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
  housing_train, housing_val, price_train, price_val, sale_mean = defs.prep_strat_set("train.csv")
  from sklearn.ensemble import GradientBoostingRegressor

  
  '''for depth in range(3,10):

    gbr = GradientBoostingRegressor(max_depth=depth,validation_fraction=0.2,n_iter_no_change = 5)
    gbr.fit(housing_train,price_train)
    print("depth:",depth)
    print_rmse(gbr,housing_train,price_train,housing_val,price_val,sale_mean)'''
  gbr = GradientBoostingRegressor(max_depth=3,validation_fraction=0.05,n_iter_no_change = 5)
  gbr.fit(housing_train,price_train)
  print_rmse(gbr,housing_train,price_train,housing_val,price_val,sale_mean)
  feature_importances = gbr.feature_importances_

  #for a,b in zip(feature_importances,attributes):
  #  print(a,b) 
if __name__=="__main__":

  main() 
