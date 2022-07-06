#!/usr/bin/python3
import math
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
def read_sets(test_path,train_path):
#read files

  test_set = pd.read_csv(test_path)
  train_set = pd.read_csv(train_path)
  return test_set,train_set

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
  for i in range(len(data.columns)):

    if data[data.columns[i]].dtype == "object":

       obj_data_list.append(data.columns[i])
  
  ordinal_encoder = OrdinalEncoder()
  data_enc = data.copy()
  for attr in obj_data_list:
    tmp = ordinal_encoder.fit_transform(data[[attr]])
    data_enc[[attr]] = tmp
  return data_enc

def find_high_corrs(data,cut_off):
#find correlation matrix

  corr_matrix = data.corr()
  print(corr_matrix.info("Column"))


def main():

  test_set,train_set = read_sets("test.csv","train.csv")
  num_cols = len(train_set.columns)
  attributes = []
  for col in train_set.columns:
    attributes.append(col)
  attributes.remove("SalePrice")
  # look for missing values
  #for attr in train_set.columns:
    #print(attr, train_set[attr].isnull().sum())
  pd.set_option('display.max_rows', None)
  #print(train_set.isnull().sum().sort_values(ascending=False))
  
  ## replace nans with na for object data
  
  train_set = clean_up_nans(train_set)

  #check null count again

  #print(train_set.isnull().sum().sort_values(ascending=False))
  # object data need to convert to numbers
  train_set_enc = encode_data(train_set)
  
#  num_data_list = list(set(train_set_enc.columns)-set(obj_data_list))

  #try a simple linear model
  from sklearn.linear_model import LinearRegression


  housing_values = train_set_enc["SalePrice"]
  housing_data = train_set_enc.drop("SalePrice", axis=1)

  def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())



  # try random forest
  from sklearn.ensemble import RandomForestRegressor
  
  # let's try normalizing data
  dat = train_set_enc.values
  min_max_scaler = preprocessing.MinMaxScaler()

  dat_scaled = min_max_scaler.fit_transform(dat)
  train_set_enc = pd.DataFrame(dat_scaled)
  housing_data = train_set_enc
 

  forest_reg = RandomForestRegressor(bootstrap=True, max_features=18,n_estimators=30)
  
 
  # let's try support vector

  from sklearn.svm import SVR

  svm_poly_reg = SVR(kernel="rbf", degree=12, C=1e7, epsilon=0.01)
  svm_poly_reg.fit(housing_data,housing_values)


  from sklearn.ensemble import VotingRegressor
  from sklearn.model_selection import train_test_split

  housing_train, housing_val, price_train, price_val = train_test_split(housing_data,housing_values, test_size = 0.2)
  
  from sklearn.ensemble import GradientBoostingRegressor

  gbr = GradientBoostingRegressor()
  #vote = VotingRegressor(estimators=[('lr',lin_reg),('rf',forest_reg),('svm',svm_poly_reg),('gbr',gbr)])
  import xgboost

  xgb = xgboost.XGBRegressor()

  vote = VotingRegressor(estimators=[('rf',forest_reg),('svm',svm_poly_reg),('gbr',gbr)])
  vote.fit(housing_train,price_train)

  vote_rmse_train = np.sqrt(mean_squared_error(price_train,vote.predict(housing_train)))
  vote_rmse_val = np.sqrt(mean_squared_error(price_val,vote.predict(housing_val)))
  print("Ensemble:",vote_rmse_train,vote_rmse_val)
  print(price_val[:5],vote.predict(housing_val)[:5])  
'''
  # implement stacking algorithm

  subset1_data, subset2_data, subset1_values, subset2_values = train_test_split(housing_train,price_train, test_size = 0.5)
  # train random_forest, svm, and gbr

  forest_reg.fit(subset1_data,subset1_values)
  svm_poly_reg.fit(subset1_data,subset1_values)
  gbr.fit(subset1_data,subset1_values)

  rf_pred = forest_reg.predict(subset2_data)
  svm_pred = svm_poly_reg.predict(subset2_data)
  gbr_pred = gbr.predict(subset2_data)

  
  new = np.stack((rf_pred,svm_pred,gbr_pred),axis=-1)
  forest_reg_stack = RandomForestRegressor()
  forest_reg_stack.fit(new,subset2_values)

  stack_rmse_train = np.sqrt(mean_squared_error(subset2_values,forest_reg_stack.predict(new)))
  
  print("Stack:",stack_rmse_train)


  rf_pred = forest_reg.predict(housing_val)
  svm_pred = svm_poly_reg.predict(housing_val)
  gbr_pred = gbr.predict(housing_val)


  new = np.stack((rf_pred,svm_pred,gbr_pred),axis=-1)


  stack_rmse_val = np.sqrt(mean_squared_error(price_val,forest_reg_stack.predict(new)))

  print("Stack:",stack_rmse_val)


  # multilayer stack
  #split data into three sets
  lay1_data, rest_data, lay1_values, rest_values = train_test_split(housing_train,price_train, test_size = 0.5)
  lay2_data, lay3_data, lay2_values, lay3_values = train_test_split(rest_data,rest_values, test_size = 0.5)
  
  # first layer: svm poly, random forest, gradient booster
  svm_poly_lay1 = SVR(kernel="rbf", degree=12, C=1e7, epsilon=0.01)
  rf_lay1 = RandomForestRegressor()
  gb_lay1 = GradientBoostingRegressor()

  svm_poly_lay1.fit(lay1_data,lay1_values)
  rf_lay1.fit(lay1_data,lay1_values)
  gb_lay1.fit(lay1_data,lay1_values)

  
  # second layer : linear regressor, random forest
  
  lay2_input = np.stack((svm_poly_lay1.predict(lay2_data),rf_lay1.predict(lay2_data),gb_lay1.predict(lay2_data)),axis=-1)
  lin_reg_lay2 = LinearRegression()
  rf_lay2 = RandomForestRegressor()

  
  lin_reg_lay2.fit(lay2_input,lay2_values)
  rf_lay2.fit(lay2_input,lay2_values)



  # send the third set through the layers
  lay3_tmp = np.stack((svm_poly_lay1.predict(lay3_data),rf_lay1.predict(lay3_data),gb_lay1.predict(lay3_data)),axis=-1) 
  lay3_input = np.stack((lin_reg_lay2.predict(lay3_tmp),rf_lay2.predict(lay3_tmp)),axis=-1)

  rf_blender = RandomForestRegressor()

  xgb.fit(lay3_input,lay3_values)

  #send the test set through

  
  lay2_test = np.stack((svm_poly_lay1.predict(housing_val),rf_lay1.predict(housing_val),gb_lay1.predict(housing_val)),axis=-1)
  lay3_test = np.stack((lin_reg_lay2.predict(lay2_test),rf_lay2.predict(lay2_test)),axis=-1)
  stack_2l_rmse_val = np.sqrt(mean_squared_error(price_val,xgb.predict(lay3_test)))

  print("2 Layer:",stack_2l_rmse_val)'''
if __name__ == "__main__":

  main() 
