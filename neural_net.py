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

def plot_train(his):
  his_ = pd.DataFrame(his.history)
  his_["loss"] = his_["loss"]
  his_["val_loss"] = his_["val_loss"]

  his_.plot(figsize=(8,5))
  plt.grid(True)
  plt.gca().set_ylim(0,.1)
  plt.title("Learning Rate")
  plt.xlabel("Epochs")
  plt.ylabel("Mean Squared Error")
  plt.show()

def main():

  housing_data = defs.read_set("train.csv")
  housing_train, housing_val, price_train, price_val, sale_mean = defs.prep_strat_set("train.csv",0.1)
 
  model = keras.models.Sequential([keras.layers.Dense(100,activation="sigmoid",input_shape=housing_train.shape[1:]),keras.layers.Dense(50,activation="sigmoid"),keras.layers.Dense(1)])
 # model = keras.models.Sequential([keras.layers.Dense(500,activation="relu",input_shape=housing_train.shape[1:]),keras.layers.Dense(100,activation="relu",input_shape=housing_train.shape[1:]),keras.layers.Dense(25,activation="relu",input_shape=housing_train.shape[1:]),keras.layers.Dense(1)])
  model.compile(loss="mean_squared_error",optimizer="sgd")

  #history = model.fit(housing_train, price_train, epochs=4000, validation_data=(housing_val,price_val))

  #plot_train(history) 
  
  #pred = model.predict(housing_val)
 
  #rmse = sale_mean *  np.sqrt(mean_squared_error(price_val,pred))
  #print("RMSE",rmse)
  fast_track = ["OverallCond","ExterCond","BsmtCond","GarageCond"] # fast track  condition scores
  #ft = pd.merge(housing_train["OverallCond"],housing_train["ExterCond"],housing_train["BsmtCond"],housing_train["GarageCond"])
  def split_wide_deep(train_set):
    df = pd.DataFrame({"OverallCond" : train_set["OverallCond"]})
    print(df.shape)
    df = df.join(pd.DataFrame({"ExterCond":train_set["ExterCond"]}),how='right')
    df = df.join(train_set["BsmtCond"])
    df = df.join(train_set["GarageCond"])
 #   print(df.info())
    housing_train_drop = train_set.drop("OverallCond",axis=1)
    housing_train_drop = housing_train_drop.drop("ExterCond",axis=1)
    housing_train_drop = housing_train_drop.drop("BsmtCond",axis=1)
    housing_train_drop = housing_train_drop.drop("GarageCond",axis=1)
    return df, housing_train_drop
  housing_train_wide, housing_train_deep = split_wide_deep(housing_train)
  housing_val_wide, housing_val_deep = split_wide_deep(housing_val)
  
  
  
  pd.set_option('display.max_rows', None) 
  input_ = keras.layers.Input(shape=[211])
  input_wide = keras.layers.Input(shape=[4])
  hidden1 = keras.layers.Dense(100, activation="sigmoid")(input_)
  hidden2 = keras.layers.Dense(100, activation="sigmoid")(hidden1) #50
  hidden3 = keras.layers.Dense(10, activation="sigmoid")(hidden2) # 10
  concat = keras.layers.concatenate([input_wide,hidden3])
  
  output = keras.layers.Dense(1)(concat)
  model =keras.models.Model(inputs=[input_wide,input_],outputs=[output])
  
  opt_list = ["sgd","RMSprop"]#"Adadelta","Adagrad","Adamax","Nadam","Ftrl"]
  # just the stochastic gradient descent seemed to work the best but feel free to try the others
  for opt in opt_list:
   if opt == "sgd":  
    model.compile(loss=["mse","mse"],optimizer=opt) 
  
    history = model.fit((housing_train_wide,housing_train_deep),price_train,epochs=2000, validation_data=((housing_val_wide,housing_val_deep),price_val))
    print(model.summary())
    

    pred = model.predict((housing_val_wide,housing_val_deep))

    rmse = sale_mean *  np.sqrt(mean_squared_error(price_val,pred))
    print("RMSE",rmse)
    plot_train(history)

if __name__=="__main__":
  main()
