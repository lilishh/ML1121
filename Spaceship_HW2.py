import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset
df = pd.read_csv("train1.csv")
df.head()
df.info()


df.drop(['Name','HomePlanet'],axis=1,inplace=True)

df.groupby('Transported').mean(numeric_only=True)


# handle missing value
df.isnull().sum() > len(df)/2 #看有哪一個欄位資料缺太多

len(df)/2


#Cabin 缺太多了 刪掉
df.drop(['Cabin'],axis=1,inplace=True)

df['Age'].isnull().value_counts()
# Age is also have some missing values
df['Age'].fillna(df['Age'].value_counts().idxmax(),inplace=True)
df['VIP'].fillna(df['VIP'].value_counts().idxmax(),inplace=True)
df['CryoSleep'].fillna(df['CryoSleep'].value_counts().idxmax(),inplace=True)
df['Destination'].value_counts().idxmax()
df['Destination'].fillna(df['Destination'].value_counts().idxmax(),inplace=True)


df = pd.get_dummies(data=df, dtype = int, columns= ['CryoSleep','Destination','VIP'])
df.head()
df.isnull().sum()

#df.drop('VIP_True', axis=1, inplace=True)
df.head()


df.corr()
X = df.drop(['Transported','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'],axis=1)
y = df['Transported']


# 機器學習
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=67)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)


from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
accuracy_score(y_test,predictions)
recall_score(y_test,predictions)
precision_score(y_test,predictions)
pd.DataFrame(confusion_matrix(y_test,predictions),columns=['Predictnot not Transported', 'Predict Transported'],index=['True not Transported','True Transported'])


# 模型匯出
import joblib
joblib.dump(lr,'Spaceship-LR.pkl',compress=3)