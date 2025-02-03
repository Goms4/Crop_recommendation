import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv(r"C:\Users\User\Downloads\crop_recommendation\crop_recommendation\Crop_recommendation.csv")  # Ensure CSV is in the same directory

# Encode categorical labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split features and target
X = df.iloc[:, :-1]  # First 7 columns
y = df.iloc[:, -1]   # Label column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and encoder
joblib.dump(model, "crop_prediction_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model and Label Encoder saved successfully!")