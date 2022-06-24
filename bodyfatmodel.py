import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

## Importing dataset as bf_data

bf_data = pd.read_csv("bodyfat.csv")
bf_data.dtype()
bf_data.head()

bf_data.describe()

bf_data.info()

# data analysis
"""
sns.jointplot(data=bf_data,x='BodyFat',y='Density')

sns.jointplot(data=bf_data,x='BodyFat',y='Age')

sns.jointplot(data=bf_data,x='BodyFat',y='Weight')

sns.jointplot(data=bf_data,x='BodyFat',y='Height')

sns.pairplot(bf_data)

sns.lmplot(data=bf_data, x='BodyFat',y='Abdomen')
sns.lmplot(data=bf_data, x='BodyFat',y='Chest')
sns.lmplot(data=bf_data, x='BodyFat',y='Ankle')
"""
# setting training and testing data

bf_data.columns


y = bf_data['BodyFat']
X = bf_data[['Weight', 'Height', 'Neck', 'Chest',
       'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']]

# splitting the data into testing AND TRAINING SETS

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training the model

from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train,y_train)

lm.coef_

# prediction

predictions = lm.predict(X_test)

#sns.jointplot(x=predictions,y=y_test)
plt.scatter(predictions,y_test,marker="1")
plt.xlabel='Predicted BodyFat'
plt.ylabel='Real BodyFat'


# evaluating the model

from sklearn import metrics

metrics.explained_variance_score(predictions,y_test)

print("MAE ",metrics.mean_absolute_error(predictions,y_test))
print("MSE ",metrics.mean_squared_error(predictions,y_test))
print("RMSE ",np.sqrt(metrics.mean_squared_error(predictions,y_test)))

sns.histplot(y_test-predictions,kde=True,bins=30)

bf_coefs = pd.DataFrame(lm.coef_,X.columns,columns=['coefficient'])
bf_coefs

import joblib
joblib.dump(lm,"bodyfat.pkl")

#model=joblib.load("bodyfat.pkl")
import pickle

#print(model.predict([154.25,67.75,36.2,93.1,85.2,94.5,59,37.3,21.9,32.0,27.4,17.1]))
pickle.dump(lm,open('lm.pkl','wb'))

lm=pickle.load(open('lm.pkl','rb'))

#print(lm.predict([154.25,67.75,36.2,93.1,85.2,94.5,59,37.3,21.9,32.0,27.4,17.1]))


