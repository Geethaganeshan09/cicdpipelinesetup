import pytest
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:/Users/geeth/Downloads/personel/cicdpipelineirissample/cicdpipelinesetup/Iris.csv")

# Prepare the data
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'my_model.joblib')

# Load the model
loaded_model = joblib.load('my_model.joblib')

def test_model_training():
    assert model is not None

def test_model_prediction():
    predictions = loaded_model.predict(X_test)
    assert len(predictions) == len(y_test)

def test_classification_report():
    predictions = loaded_model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    assert 'accuracy' in report

def test_confusion_matrix():
    predictions = loaded_model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)
    assert matrix.shape == (3, 3)

def test_single_prediction():
    test_plant = {
        'SepalLengthCm': 5.1,
        'SepalWidthCm': 3.5,
        'PetalLengthCm': 1.4,
        'PetalWidthCm': 0.2
    }
    plant_df = pd.DataFrame([test_plant])
    prediction = loaded_model.predict(plant_df)
    assert prediction[0] in df['Species'].unique()

if __name__ == "__main__":
    pytest.main()