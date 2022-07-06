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

  housing_data, housing_prices = defs.prep_sets("test.csv","train.csv") 
  
  #try a polynomial model
  
  housing_train, housing_val, price_train, price_val = train_test_split(housing_data,housing_prices, test_size = 0.2)
  poly_reg = Pipeline([("poly_features",PolynomialFeatures(degree=3,interaction_only=False,include_bias=False)),("lin_reg",LinearRegression()),])
  scores = cross_val_score(poly_reg, housing_train, price_train,scoring="neg_mean_squared_error",cv=5)
  poly_rmse_scores = np.sqrt(-scores)
  
  defs.display_scores(poly_rmse_scores)
  
  poly_reg.fit(housing_train,price_train)
   
  #plot_learning(poly_reg,housing_data,housing_prices)
  
  housing_train_pred = poly_reg.predict(housing_train)
  housing_val_pred = poly_reg.predict(housing_val)
  
  poly_mse_train = np.sqrt(mean_squared_error(price_train,housing_train_pred))
  poly_mse_val = np.sqrt(mean_squared_error(price_val,housing_val_pred))
  print("Poly mean squared error: ", "train:",poly_mse_train,"validation:",poly_mse_val)  
  print(poly_reg.predict(housing_val)[:10])
  print(price_val[:10])

if __name__ == "__main__":

  main()
