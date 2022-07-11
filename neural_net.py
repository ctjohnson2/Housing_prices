#!/usr/bin/python3

import defs
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from warnings import simplefilter
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
simplefilter(action='ignore',category=FutureWarning)

def main():

  housing_data = defs.read_set("train.csv")
  housing_train, housing_val, price_train, price_val, sale_mean = defs.prep_strat_set("train.csv")
  print(housing_train.shape[1:])
  model = keras.models.Sequential([keras.layers.Dense(100,activation="relu",input_shape=housing_train.shape[1:]),keras.layers.Dense(50,activation="relu"),keras.layers.Dense(1)])
  model.compile(loss="mean_squared_error",optimizer="sgd")

  history = model.fit(housing_train, price_train, epochs=1000, validation_data=(housing_val,price_val))

  his_ = pd.DataFrame(history.history)
  his_["loss"] = his_["loss"]
  his_["val_loss"] = his_["val_loss"]
  print(his_)
  his_.plot(figsize=(8,5))
  plt.grid(True)
  plt.gca().set_ylim(0,1)
  plt.show()
  pred = model.predict(housing_val)
 
  rmse = sale_mean *  np.sqrt(mean_squared_error(price_val,pred))
  print("RMSE",rmse)
if __name__=="__main__":
  main()
