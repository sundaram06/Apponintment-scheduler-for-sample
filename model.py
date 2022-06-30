import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

sample=pd.read_csv(r'E:\Pro 69\pro\final_data.csv')
sample.info()

sample=sample.drop({'Agent Arrival Time (range) HH:MM','Unnamed: 0','Patient ID','pincode','patient location','Latitudes and Longitudes (Patient)','Latitudes and Longitudes (Agent)','Latitudes and Longitudes (Diagnostic Center)','Age','Gender','Time slot','Availabilty time (Patient)','Test Booking Time HH:MM','Test Booking Date','Sample Collection Date'},axis=1)
sample

duplicate = sample.duplicated()
sum(duplicate)

lb=LabelEncoder()
sample['Diagnostic Centers']=lb.fit_transform(sample['Diagnostic Centers'])
sample['Test name']=lb.fit_transform(sample['Test name'])
sample['Sample']=lb.fit_transform(sample['Sample'])
sample['Way Of Storage Of Sample']=lb.fit_transform(sample['Way Of Storage Of Sample'])

X=sample.iloc[:,:9]
y=sample['Exact Arrival Time MM']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X.info()
# Fitting Random Forest Regression to the dataset

mod = RandomForestRegressor(n_estimators = 10, random_state = 40)
mod.fit(x_train,y_train)
pre=mod.predict(x_test)
mean_squared_error(y_test, pre)
r2_score(y_test,pre)

pre1=mod.predict(x_train)

# Error on test dataset
mean_squared_error(y_train, pre1)
r2_score(y_train,pre1)


pickle.dump(mod, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
