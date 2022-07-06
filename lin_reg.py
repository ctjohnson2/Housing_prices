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
def plot_learning(model,x_train,x_val,y_train,y_val,mean_):

  
  train_err, val_err,sam_size = [],[],[]
  for i in range(1,len(x_train),100):
    sam_size.append(i)
    model.fit(x_train[:i],y_train[:i])
    y_train_predict = model.predict(x_train[:i])
    y_val_predict = model.predict(x_val)
    train_err.append(mean_squared_error(y_train[:i]*mean_,y_train_predict[:i]*mean_))
    val_err.append(mean_squared_error(y_val*mean_,y_val_predict*mean_))
  plt.axis([0,1500,0,100000])
  plt.ylabel("RMSE")
  plt.xlabel("Data Size")
  plt.title("Learning Rate")
  
  
  plt.plot(sam_size,np.sqrt(train_err),"r-+",linewidth=2,label="Training Set")
  plt.plot(sam_size,np.sqrt(val_err),"b-",linewidth=3,label="Validation Set")
  plt.legend()
  plt.show()

def main():
  
  
  housing_data = defs.read_set("train.csv")
  housing_train, housing_val, price_train, price_val, sale_mean = defs.prep_strat_set("train.csv")
  
  '''lin_reg = LinearRegression()
   
  #plot_learning(lin_reg,housing_train, housing_val, price_train, price_val, sale_mean)
  lin_reg.fit(housing_train,price_train)
  housing_train_pred = lin_reg.predict(housing_train)
  housing_val_pred = lin_reg.predict(housing_val)
  from sklearn.metrics import mean_squared_error
  lin_mse_train = sale_mean*np.sqrt(mean_squared_error(price_train,housing_train_pred))
  lin_mse_val = sale_mean*np.sqrt(mean_squared_error(price_val,housing_val_pred))
  print("lin mean squared error: ", "train:",lin_mse_train,"validation:",lin_mse_val)  
  '''
  from sklearn.linear_model import Ridge
  ridge_reg = Ridge(alpha=1,solver="cholesky")

  plot_learning(ridge_reg,housing_train, housing_val, price_train, price_val, sale_mean)
  ridge_reg.fit(housing_train,price_train)
  housing_train_pred = ridge_reg.predict(housing_train)
  housing_val_pred = ridge_reg.predict(housing_val)
  
  lin_mse_train = sale_mean * np.sqrt(mean_squared_error(price_train,housing_train_pred))
  lin_mse_val = sale_mean *np.sqrt(mean_squared_error(price_val,housing_val_pred))
  print("Ridge mean squared error: ", "train:",lin_mse_train,"validation:",lin_mse_val)
  print(sale_mean * price_val[:5],sale_mean * housing_val_pred[:5])
  
  x_ = range(len(housing_val_pred))
  plt.bar(x_,sale_mean*housing_val_pred,color='red',alpha=0.5,label='prediction')
  plt.bar(x_,sale_mean*price_val,color='blue',alpha=0.5,label='actual')
  plt.ylabel("Prices in $")
  plt.title("Validation Results")
  plt.text(0,4e5,"RMSE= "+str(lin_mse_val))
  plt.legend()
  plt.show()


  bins_ = range(0,1000000,5000)
  
  housing_val_pred = pd.DataFrame(data=sale_mean*housing_val_pred,columns=["Pricepred"])
  binned = pd.cut(housing_val_pred["Pricepred"],bins=bins_).apply(lambda x: x.left) 
  print(binned)
  lin_mse_val_bin = np.sqrt(mean_squared_error(sale_mean*price_val,binned))
  print(lin_mse_val_bin)

if __name__ == "__main__":

  main()
