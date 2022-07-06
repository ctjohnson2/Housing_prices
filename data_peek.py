#!/usr/bin/python3

import defs
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)
def main():

  test, housing_data = defs.read_sets("test.csv","train.csv")
  attributes = []
  for col in housing_data.columns:
    attributes.append(col)

  housing_data = defs.clean_up_nans(housing_data)

  sale_min = housing_data["SalePrice"].min()
  sale_max = housing_data["SalePrice"].max()

  housing_prices = housing_data["SalePrice"]
  housing_data = housing_data.drop("SalePrice",axis=1)

  sale_mean = housing_prices.mean()
  housing_prices = housing_prices/housing_prices.mean()
  housing_train, housing_val, price_train, price_val = train_test_split(housing_data,housing_prices, test_size = 0.2)
  
  from sklearn.model_selection import StratifiedShuffleSplit
  split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
  print(split.split(housing_data,housing_data["Neighborhood"]))
  for train_ind, test_ind in split.split(housing_data,housing_data["Neighborhood"]):
    tr = train_ind
    ts = test_ind
    strat_train_set = housing_data.loc[train_ind]
    strat_test_set = housing_data.loc[test_ind]
  print(tr)
  print(ts)
  figure, axis = plt.subplots(2,3)
  
  axis[0,1].hist(housing_data["Functional"],color='red',density=True,bins=5)
  axis[0,1].hist(housing_train["Functional"],color='blue',density=True,alpha =0.5,bins=5)
  axis[0,1].set_title("Functional")
  axis[0,2].hist(housing_data["OverallCond"],color='red',density=True,bins=24)
  axis[0,2].hist(housing_train["OverallCond"],color='blue',density=True,alpha = 0.5,bins=24)
  axis[0,2].set_title("OverallCond")
  axis[1,0].hist(housing_data["Neighborhood"],color='red',density=True,bins=24)
  axis[1,0].hist(housing_train["Neighborhood"],color='blue',density=True,alpha = 0.5,bins=24)
  axis[1,0].set_title("Neighborhoods")
  plt.xticks(rotation=90)
  axis[0,0].hist(housing_data["MSSubClass"],color='red',density = True, bins=16)
  axis[0,0].hist(housing_train["MSSubClass"],color='blue',density=True,alpha = 0.5,bins=16)
  axis[0,0].set_title("MSSubClass")
  plt.show()

  # This shows train_test_split samples Neighborhood and MSSubClass according
  # to their distributions

  figure, axis_new = plt.subplots(2,3)
  axis_new[0,1].hist(housing_data["Functional"],color='red',density=True,bins=5)
  axis_new[0,1].hist(housing_val["Functional"],color='blue',density=True,alpha =0.5,bins=5)
  axis_new[0,1].set_title("Functional")
  axis_new[0,2].hist(housing_data["OverallCond"],color='red',density=True,bins=24)
  axis_new[0,2].hist(housing_val["OverallCond"],color='blue',density=True,alpha = 0.5,bins=24)
  axis_new[0,2].set_title("OverallCond")
  axis_new[1,0].hist(housing_data["Neighborhood"],color='red',density=True,bins=24)
  axis_new[1,0].hist(housing_val["Neighborhood"],color='blue',density=True,alpha = 0.5,bins=24)
  axis_new[1,0].set_title("Neighborhoods")
  plt.xticks(rotation=90)
  axis_new[0,0].hist(housing_data["MSSubClass"],color='red',density = True, bins=16)
  axis_new[0,0].hist(housing_val["MSSubClass"],color='blue',density=True,alpha = 0.5,bins=16)
  axis_new[0,0].set_title("MSSubClass")
  plt.show()

  figure, axis_new = plt.subplots(2,3)
  axis_new[0,1].hist(strat_train_set["Functional"],color='red',density=True,bins=5)
  axis_new[0,1].hist(strat_test_set["Functional"],color='blue',density=True,alpha =0.5,bins=5)
  axis_new[0,1].set_title("Functional")
  axis_new[0,2].hist(strat_train_set["OverallCond"],color='red',density=True,bins=24)
  axis_new[0,2].hist(strat_test_set["OverallCond"],color='blue',density=True,alpha = 0.5,bins=24)
  axis_new[0,2].set_title("OverallCond")
  axis_new[1,0].hist(strat_train_set["Neighborhood"],color='red',density=True,bins=24)
  axis_new[1,0].hist(strat_test_set["Neighborhood"],color='blue',density=True,alpha = 0.5,bins=24)
  axis_new[1,0].set_title("Neighborhoods")
  plt.xticks(rotation=90)
  axis_new[0,0].hist(strat_train_set["MSSubClass"],color='red',density = True, bins=16)
  axis_new[0,0].hist(strat_test_set["MSSubClass"],color='blue',density=True,alpha = 0.5,bins=16)
  axis_new[0,0].set_title("MSSubClass")
  plt.show()
if __name__=="__main__":
  main()
