#!/usr/bin/python3

import defs
import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def main():

  housing_data = defs.prep_sets("test.csv","train.csv")
  attributes = []
  for col in housing_data.columns:
    attributes.append(col)

  sale_min = housing_data["SalePrice"].min()
  sale_max = housing_data["SalePrice"].max()

  dat = housing_data.values
  min_max_scaler = preprocessing.MinMaxScaler()

  dat_scaled = min_max_scaler.fit_transform(dat)
  housing_data = pd.DataFrame(dat_scaled,columns = attributes)

  housing_prices = housing_data["SalePrice"]
  housing_data = housing_data.drop("SalePrice",axis=1)


  housing_prices = housing_prices*(sale_max - sale_min)+sale_min*np.ones(len(housing_prices))
  sale_mean = housing_prices.mean()
  housing_prices = housing_prices/housing_prices.mean() 
  housing_train, housing_val, price_train, price_val = train_test_split(housing_data,housing_prices, test_size = 0.2)
  
  look_for_best_poly = True
  
  if look_for_best_poly==True:
    Cvals =[1,1e1,1e2,1e3,1e4]
    es =[1e-1]
    val_low =1e5
    C_low,or_low = 0,0
    for c in Cvals:
      for e in es:
        for poly_or in range(2,10):
          svm_poly_reg = SVR(kernel="poly", degree=poly_or, C=c, epsilon=e)
          svm_poly_reg.fit(housing_train,price_train)
          rmse_train = sale_mean* np.sqrt(mean_squared_error(price_train,svm_poly_reg.predict(housing_train)))
          rmse_val = np.sqrt(mean_squared_error(price_val,svm_poly_reg.predict(housing_val)))*sale_mean
          if rmse_val < val_low:
             val_low = rmse_val
             or_low = poly_or
             C_low = c
          print("C=",c,"epsilon=",e,"poly",poly_or)
          print("RMSE:","Train=",rmse_train,"Val=",rmse_val)

    print("C_Low=",C_low,"Order",or_low,"rmse",val_low)

  # poly 4 C=1e4 seemed to give the best validation

  look_for_best_rbf = True
  if look_for_best_rbf:

     gam = [100,10,1,1e-1,1e-2,1e-3,1e-4]
     Cvals =[1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8]
     C_low,gam_low =0,0
     val_low = 1e5
     for c in Cvals:
       for g in gam:

         svm_rbf_reg=SVR(kernel="rbf",gamma=g,C=c)
         svm_rbf_reg.fit(housing_train,price_train)
         rmse_train = sale_mean * np.sqrt(mean_squared_error(price_train,svm_rbf_reg.predict(housing_train)))
         rmse_val = sale_mean * np.sqrt(mean_squared_error(price_val,svm_rbf_reg.predict(housing_val)))
  
         if rmse_val < val_low:
             val_low = rmse_val
             gam_low = g
             C_low = c
         print("C=",c,"gamma=",g)
         print("RMSE:","Train=",rmse_train,"Val=",rmse_val)

     print("C_Low=",C_low,"Gamma",gam_low,"rmse",val_low)

  # best C=1e6 gam =0.01
if __name__=="__main__":

  main()
