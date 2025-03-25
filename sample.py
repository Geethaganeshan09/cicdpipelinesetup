

import pandas as pd
import joblib
df = pd.read_csv("Iris.csv")
df.head()

df['Species'].unique()


df.isna().sum()

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

test_plant = {
    'SepalLengthCm': 5.1,
    'SepalWidthCm': 3.5,
    'PetalLengthCm': 1.4,
    'PetalWidthCm': 0.2
}

plant_df = pd.DataFrame([test_plant])
plant_df

model.predict(plant_df)

joblib.dump(model, 'my_model.joblib')



