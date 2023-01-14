import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle

df=pd.read_csv("Clean_Dataset.csv")
df.drop(columns=["Unnamed: 0"],inplace=True)

num_col=df.dtypes[df.dtypes!="object"].index
cat_col=df.dtypes[df.dtypes=="object"].index

def out_treat(x):
    x=x.clip(upper=x.clip(x.quantile(0.99)))
    return(x)

df[num_col]=df[num_col].apply(out_treat)

lb=LabelEncoder()
for i in cat_col:
    df[i]=lb.fit_transform(df[i])

x=df.drop(columns=["price","flight"])
y=df["price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

lin_reg= LinearRegression()
dec_tree = DecisionTreeRegressor(criterion="squared_error",max_depth=5,min_samples_split=15,min_samples_leaf=10)
rand_for=RandomForestRegressor(criterion='squared_error',n_estimators=100,max_depth=10,min_samples_split=15)
xgb=XGBRegressor(n_estimators=100,max_depth=4,reg_lambda=.2,eta=0.3,eval_metric='rmse',gamma=0.5,objectives='reg:squarederror',
                random_state=0,reg_alpha=0)

addt=DecisionTreeRegressor(criterion="squared_error",max_depth=5,min_samples_split=15,min_samples_leaf=10)
ada_reg=AdaBoostRegressor(base_estimator=addt,n_estimators=100)

lin_reg = lin_reg.fit(x_train,y_train)
dec_tree=dec_tree.fit(x_train,y_train)
rand_for=rand_for.fit(x_train,y_train)
xgb=xgb.fit(x_train,y_train)
ada_reg=ada_reg.fit(x_train,y_train)

pickle.dump(lin_reg,open('lin_model.pkl','wb'))
pickle.dump(dec_tree,open('dec_tree_model.pkl','wb'))
pickle.dump(rand_for,open('rand_for_model.pkl','wb'))
pickle.dump(xgb,open('xgb_model.pkl','wb'))
pickle.dump(ada_reg,open('ada_reg_model.pkl','wb'))