import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. LOAD THE BRAIN
# We load the dictionary we saved earlier
model_package = joblib.load('breast_cancer_rf_model.pkl')

# Extract the actual model and the feature names
model = model_package['model']
feature_names = model_package['features']

data  = load_breast_cancer()

print(f"✅ Model Loaded: {model_package['metadata']['description']}")
print(f"   Accuracy on Test Data was: {model_package['metadata']['accuracy']:.2%}")

# 2. THE NEW PATIENT (Simulation)
# This represents data coming from a frontend or API.
# It MUST match the exact shape (30 features) the model was trained on.
# Here is a fake patient data point (random numbers for demo):
new_patient_data = np.random.rand(1, 30) 

print("\n--- New Patient Data ---")
for name, value in zip(feature_names, new_patient_data[0]):
    print(f"{name}: {value:.4f}")

# 3. MAKE PREDICTION
prediction = model.predict(new_patient_data)
probability = model.predict_proba(new_patient_data)

# 4. OUTPUT
class_name = data.target_names[prediction[0]] # 'malignant' or 'benign'
confidence = np.max(probability) * 100

print("\n--- Diagnostic Result ---")
print(f"Prediction: {class_name.upper()}")
print(f"Confidence: {confidence:.2f}%")

# NICHE TRICK:
# If you forgot which feature is which column, check the saved metadata:
# print("Features expected:", feature_names)