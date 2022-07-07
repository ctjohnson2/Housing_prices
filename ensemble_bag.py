#!/usr/bin/python3

import defs
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from warnings import simplefilter
import matplotlib.pyplot as plt

simplefilter(action='ignore',category=FutureWarning)

def print_rmse(model,train,price_train,val,price_val,mean_):
#  mean_=1
  vote_rmse_train = mean_*np.sqrt(mean_squared_error(price_train,model.predict(train)))
  vote_rmse_val = mean_*np.sqrt(mean_squared_error(price_val,model.predict(val)))
  print(str(model),":",vote_rmse_train,vote_rmse_val)
def main():
 
  housing_data = defs.read_set("train.csv")
  housing_train, housing_val, price_train, price_val, sale_mean = defs.prep_strat_set("train.csv")

  lin_reg = LinearRegression()
  from sklearn.linear_model import Ridge
  ridge_reg = Ridge(alpha=1,solver="cholesky")
  svm_poly_reg = SVR(kernel="poly", degree=4, C=1e3)
  svm_rbf_reg = SVR(kernel="rbf", degree=12, C=1e6, epsilon=0.01) 
  forest_reg = RandomForestRegressor(bootstrap=True,max_features=20,n_estimators=70)
  from sklearn.ensemble import GradientBoostingRegressor

  gbr = GradientBoostingRegressor(max_depth=3,n_iter_no_change=5,validation_fraction=0.05)
  import xgboost
  xgb = xgboost.XGBRegressor()
  
  ridge_reg.fit(housing_train,price_train)
  gbr.fit(housing_train,price_train)
  xgb.fit(housing_train,price_train)
  forest_reg.fit(housing_train,price_train)
  vote = VotingRegressor(estimators=[('lr',ridge_reg),('gbr',gbr),('fr',forest_reg)])
  
  print("Fitting model")
  
  vote.fit(housing_train,price_train)
 
  
  print_rmse(ridge_reg,housing_train,price_train,housing_val,price_val,sale_mean)
  print_rmse(gbr,housing_train,price_train,housing_val,price_val,sale_mean)
  print_rmse(forest_reg,housing_train,price_train,housing_val,price_val,sale_mean)
  print_rmse(vote,housing_train,price_train,housing_val,price_val,sale_mean) 
  
  housing_val_pred = vote.predict(housing_val)
  lin_mse_val =  np.sqrt(mean_squared_error(housing_val_pred,price_val))*sale_mean
  
  # plot learning rates
  models = [ridge_reg,gbr,forest_reg,vote]
  figure, axis = plt.subplots(2,3)
  cx,cy =0,0
  for model in models:
    train_err, val_err,sam_size = [],[],[]
    for i in range(20,len(housing_train),100):
      sam_size.append(i)
      model.fit(housing_train[:i],price_train[:i])
      y_train_predict = model.predict(housing_train[:i])
      y_val_predict = model.predict(housing_val)
      train_err.append(mean_squared_error(price_train[:i],y_train_predict[:i]))
      val_err.append(mean_squared_error(price_val,y_val_predict))
    axis[cx,cy].plot(sam_size,sale_mean*np.sqrt(train_err),color="red")
    axis[cx,cy].plot(sam_size,sale_mean*np.sqrt(val_err),color="blue")
    if cx==cy==0:
      title_="Ridge Regression"
    elif cx ==0 and cy ==1:
      title_="Gradient Boost"
    elif cx ==0 and cy ==2:
      title_="Random Forest"
    elif cy ==0 and cx == 1:
      title_="Vote Ensemble"
    axis[cx,cy].set_title(title_)
    cy+=1
    if cy ==3:
      cy =0
      cx+=1
    axis[1,1].set_axis_off()
    axis[1,2].set_axis_off()
  plt.show()

  # plot hist of prices predicted versus sold
  x_ = range(len(housing_val_pred))
  plt.bar(x_,sale_mean*housing_val_pred,color='red',alpha=0.5,label='prediction')
  plt.bar(x_,sale_mean*price_val,color='blue',alpha=0.5,label='actual')
  plt.ylabel("Prices in $")
  plt.title("Validation Results")
  plt.text(0,4e5,"RMSE= "+str(lin_mse_val))
  plt.legend()
  plt.show()


  
  housing_val_total = housing_val.join(price_val)
  #print(housing_data["Neighborhood"].unique())
  neighborhoods=['CollgCr', 'Veenker', 'Crawfor', 'NoRidge','Mitchel','Somerst', 'NWAmes','OldTown', 'BrkSide', 'Sawyer', 'NridgHt','NAmes', 'SawyerW', 'IDOTRR','MeadowV', 'Edwards', 'Timber', 'Gilbert','StoneBr', 'ClearCr', 'NPkVill','Blmngtn', 'BrDale', 'SWISU','Blueste']
  figure, axis = plt.subplots(5,6)
  cx,cy=0,0
  for col in housing_val_total.columns:
    for hood in neighborhoods:
      if col == (hood,):
        print(col) 
        df_new = housing_val_total.loc[housing_val_total[col] == 1.0]
        if len(df_new)!=0:
          price_new = df_new["SalePrice"]
          df_new = df_new.drop("SalePrice",axis=1) 
          pred = vote.predict(df_new)
          lin_mse_val =  np.sqrt(mean_squared_error(pred,price_new))*sale_mean
          print("hood:",hood,lin_mse_val)
          x_ = range(len(pred))
          if cx == cy and cx ==0:
            axis[cx,cy].bar(x_,sale_mean*pred,color='red',alpha = 0.5,label='prediction')
            axis[cx,cy].bar(x_,sale_mean*price_new,color='blue',alpha = 0.5,label = 'price')
          else:
            axis[cx,cy].bar(x_,sale_mean*pred,color='red',alpha = 0.5)
            axis[cx,cy].bar(x_,sale_mean*price_new,color='blue',alpha = 0.5)
          axis[cx,cy].axis([0,len(x_),0,500000])
          axis[cx,cy].set_title(hood, fontsize=7)
          axis[cx,cy].text(0,4e5,"RMSE= "+str(lin_mse_val),fontsize=5)
          axis[cx,cy].set_xticks([])
          axis[cx,cy].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
          
          
        else:
          print("No validation for ",hood)
          
        cy +=1
        if cy ==6:
           cy = 0
           cx+=1
  figure.legend()
  plt.yticks(fontsize=1)
  plt.show()    
if __name__=="__main__":

  main()
