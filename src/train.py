import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('data/students.csv')

# Encode categorical columns
le = LabelEncoder()

categorical_cols = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('math score', axis=1)

y = df['math score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy score
score = r2_score(y_test, predictions)

print("R2 Score:", score)

# Save model
pickle.dump(model, open('model/model.pkl', 'wb'))

print("Model saved successfully!")