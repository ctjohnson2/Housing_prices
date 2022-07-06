#!/usr/bin/python3
import math
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn import preprocessing

def read_sets(test_path,train_path):
#read files

  test_set = pd.read_csv(test_path)
  train_set = pd.read_csv(train_path)
  return test_set,train_set
def read_set(train_path):
  train_set = pd.read_csv(train_path)
  return train_set
def clean_up_nans(data):

  data["PoolQC"].fillna(value = "na", inplace=True)
  data["MiscFeature"].fillna(value = "na", inplace=True)
  data["Alley"].fillna(value = "na", inplace=True)
  data["Fence"].fillna(value = "na", inplace=True)
  data["FireplaceQu"].fillna(value = "na", inplace=True)
  data["GarageCond"].fillna(value = "na", inplace=True)
  data["GarageType"].fillna(value = "na", inplace=True)
  data["GarageFinish"].fillna(value = "na", inplace=True)
  data["GarageQual"].fillna(value = "na", inplace=True)
  data["BsmtFinType2"].fillna(value = "na", inplace=True)
  data["BsmtExposure"].fillna(value = "na", inplace=True)
  data["BsmtQual"].fillna(value = "na", inplace=True)
  data["BsmtCond"].fillna(value = "na", inplace=True)
  data["BsmtFinType1"].fillna(value = "na", inplace=True)
  data["MasVnrType"].fillna(value = "na", inplace=True)

  # seems LotFrontage nans are some sort of error, likely could do better but
  # for now just replace these with a median
  data["LotFrontage"].fillna(value = data["LotFrontage"].median(),
  inplace = True)
  data["GarageYrBlt"].fillna(value = data["GarageYrBlt"].median(),
  inplace = True)
  data["MasVnrArea"].fillna(value = data["MasVnrArea"].median(),
  inplace = True)
  data["Electrical"].fillna(value = "FuseA",
  inplace = True)
  
  
  
  return data
def encode_data(data):

  obj_data_list = []
  hot_list = ["Exterior1st", "Exterior2nd", "RoofMatl", "RoofStyle","HouseStyle", "BldgType", "Condition2", "Condition1", "Neighborhood","LotConfig", "MSZoning", "MasVnrType", "Foundation", "Heating","GarageType", "Fence", "MiscFeature", "SaleType"]
  one_code = []
  rest = []
  for attr in data.columns:

    if hot_list.count(attr) ==0:
      if data[attr].dtype == "object":
        one_code.append(attr)
      else:
        rest.append(attr)
  data_new = data.copy()

  
  for obj_ in hot_list:
    #print(obj_)
    cat_encoder = OneHotEncoder()
    hot_enc = cat_encoder.fit_transform(data[[obj_]])
    #print(obj_)
    #print(cat_encoder.categories_)
    enc_df = pd.DataFrame(data=hot_enc.toarray(),columns=cat_encoder.categories_)
    data_new = data_new.join(enc_df, how="right",lsuffix="_"+obj_)
    data_new = data_new.drop(obj_,axis=1) 
 

  ordinal_encoder = OrdinalEncoder()
  data_enc = data_new.copy()
  for attr in one_code:
    tmp = ordinal_encoder.fit_transform(data_new[[attr]])
    data_enc[[attr]] = tmp
  min_max_scaler = preprocessing.MinMaxScaler()
  for attr in rest:
    if attr!="SalePrice":
      dat_scaled = min_max_scaler.fit_transform(data_enc[[attr]])
      data_enc[[attr]] = dat_scaled
  
    
  return data_enc

def prep_sets(test,train):

  test_set,train_set = read_sets(test,train)
   
  pd.set_option('display.max_rows', None)

  ## replace nans with na for object data
  train_set = clean_up_nans(train_set)

  train_set_enc = encode_data(train_set)
  housing_data = train_set_enc 
  
  return housing_data
def prep_strat_set(train):

  train_dat = read_set(train)

  train_dat = clean_up_nans(train_dat)
  
  from sklearn.model_selection import StratifiedShuffleSplit
  split = StratifiedShuffleSplit(n_splits=1,test_size=0.1, random_state=42)

  for train_ind, test_ind in split.split(train_dat,train_dat["Neighborhood"]):
    train_list = train_ind
    test_list = test_ind

  housing_data = encode_data(train_dat)
 
  
  housing_prices = housing_data["SalePrice"]
  housing_data = housing_data.drop("SalePrice",axis=1)

  sale_mean = housing_prices.mean()
  housing_prices = housing_prices/housing_prices.mean()

  housing_train, price_train = housing_data.iloc[train_list],housing_prices.iloc[train_list]
  housing_val, price_val = housing_data.iloc[test_list], housing_prices.iloc[test_list]
  return housing_train, housing_val, price_train, price_val, sale_mean
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
