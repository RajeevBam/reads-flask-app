
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.utils.vis_utils import plot_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("plotly_flask_tutorial/ExplainerDashboard/files/electrochemicaldata-2.csv")
data = data.dropna()

X = data[data.columns[4:11]]
y = data[data.columns[14:15]]
y = y.values.flatten()




col_list = X.columns.to_list()

from sklearn.preprocessing import MinMaxScaler

# fit scaler on all data
norm = MinMaxScaler().fit(X)

# transform all data
X = norm.transform(X)

X = pd.DataFrame(X, columns = col_list)


model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import types
def predict_proba(self, X):
    pred = self.predict(X)
    return np.array([1-pred, pred]).T
model.predict_proba = types.MethodType(predict_proba, model)

print("X",len(X))
print("y",len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=52)

# X_train, y_train, X_test, y_test = titanic_survive()

print(X_test)
def init_explainer_dashboard(server):
    # model = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
    explainer = ClassifierExplainer(model, X_test, y_test,
                               labels=['Not survived', 'Survived'])
    db=ExplainerDashboard(explainer,server=server, routes_pathname_prefix="/expalinerdashboard/")
    return db.server


