import joblib
model_pretrained = joblib.load('Spaceship-LR.pkl')
import pandas as pd

#for submission
df_test = pd.read_csv("test1.csv")
df_test.drop(['Name','HomePlanet'], axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)

df_test.isnull().sum()

df_test['Age'].fillna(df_test['Age'].value_counts().idxmax(), inplace=True)
df_test['VIP'].fillna(df_test['VIP'].value_counts().idxmax(), inplace=True)
df_test['CryoSleep'].fillna(df_test['CryoSleep'].value_counts().idxmax(), inplace=True)
df_test['Destination'].fillna(df_test['Destination'].value_counts().idxmax(), inplace=True)

df_test.info()

df_test = pd.get_dummies(data=df_test, dtype = int, columns=['CryoSleep','Destination','VIP'])

#df_test.drop('VIP_True', axis=1, inplace=True)
df_test.drop('RoomService', axis=1, inplace=True)
df_test.drop('FoodCourt', axis=1, inplace=True)
df_test.drop('ShoppingMall', axis=1, inplace=True)
df_test.drop('Spa', axis=1, inplace=True)
df_test.drop('VRDeck', axis=1, inplace=True)

predictions2 = model_pretrained.predict(df_test)
predictions2

#press submit file
forSubmissionDF = pd.DataFrame(columns = ['PassengerId', 'Transported'])
forSubmissionDF['PassengerId'] = df_test['PassengerId']
forSubmissionDF['Transported'] = predictions2
forSubmissionDF

forSubmissionDF.to_csv('Spaceship_for_Submission.csv', index=False)
