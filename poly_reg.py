#!/usr/bin/python3

import defs
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, SVC
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)

def main():

  
  
  #try a polynomial model
  
  housing_data = defs.read_set("train.csv")
  housing_train, housing_val, price_train, price_val, sale_mean = defs.prep_strat_set("train.csv")

  poly_reg = Pipeline([("poly_features",PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)),("lin_reg",LinearRegression()),])
  scores = cross_val_score(poly_reg, housing_train, price_train,scoring="neg_mean_squared_error",cv=5)
  poly_rmse_scores = np.sqrt(-scores)
  
  defs.display_scores(poly_rmse_scores*sale_mean)
  
  poly_reg.fit(housing_train,price_train)
   
  #plot_learning(poly_reg,housing_data,housing_prices)
  
  housing_train_pred = poly_reg.predict(housing_train)
  housing_val_pred = poly_reg.predict(housing_val)
  
  poly_mse_train = sale_mean*np.sqrt(mean_squared_error(price_train,housing_train_pred))
  poly_mse_val = sale_mean*np.sqrt(mean_squared_error(price_val,housing_val_pred))
  print("Poly mean squared error: ", "train:",poly_mse_train,"validation:",poly_mse_val)  
  print(poly_reg.predict(housing_val)[:10])
  print(price_val[:10])
  # seems even just a 2nd degree poly overfits
if __name__ == "__main__":

  main()
