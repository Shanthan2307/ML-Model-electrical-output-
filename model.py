import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
#load csv.file
df=pd.read_csv("ccpp.csv")
#after obervation of heatmap and pairplots
# AT AND PE are  inversely  and highly corr(-0.95)
# V AND PE are  inversely  and highly corr(-0.87)
# Ap and Pe are moderately coor (0.52)
# Rh and PE are weakly coor (0.39)

#selecting the independent and dependent variables
x=df[['AT','V','AP','RH']]
y=df['PE']

#spliting the data into train and test set.
train_x,test_x,train_y,test_y = train_test_split(x,y, test_size=0.30, random_state=5)

#feature scaling
t_x=np.asanyarray(train_x)
t_y=np.asanyarray(train_y)

#instantiate the model
regr=linear_model.LinearRegression()

#data fitting the model
regr.fit(t_x,t_y)

#take pickle file  of our model
pickle.dump(regr,open("model.pkl","wb"))