from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data=load_iris()
x=pd.DataFrame(data.data, columns=data.feature_names)
y=data.target
print("dataset info :")
print(x.describe())
print("target classes :",data.target_names)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print("accuracy score :",accuracy_score(y_test,y_pred))

scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)
x_train_scaled,x_test_scaled,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
knn_scaled=KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(x_train_scaled,y_train)
y_pred_scaled=knn_scaled.predict(x_test_scaled)
print ("accuracy with min-max scaling :",accuracy_score(y_test,y_pred_scaled))

scaler=StandardScaler()
x_scaled_std=scaler.fit_transform(x)
x_train_scaled_std,x_test_scaled_std,y_train,y_test=train_test_split(x_scaled_std,y,test_size=0.2,random_state=42)
knn_scaled_std=KNeighborsClassifier(n_neighbors=5)
knn_scaled_std.fit(x_train_scaled_std,y_train)
y_pred_scaled_std=knn_scaled_std.predict(x_test_scaled_std)
print("accuracy with standard scaling :",accuracy_score(y_test,y_pred_scaled_std))
