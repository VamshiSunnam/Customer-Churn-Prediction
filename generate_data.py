import pandas as pd
import numpy as np
import os

# Define the number of samples
num_samples = 1000

# Generate synthetic data
data = {
    'gender': np.random.choice(['Male', 'Female'], size=num_samples),
    'age': np.random.randint(18, 70, size=num_samples),
    'tenuremonths': np.random.randint(1, 72, size=num_samples),
    'monthly_charges': np.random.uniform(20, 120, size=num_samples).round(2),
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples, p=[0.5, 0.3, 0.2]),
    'churn': np.random.choice(['Yes', 'No'], size=num_samples, p=[0.26, 0.74])
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create the data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Save the DataFrame to a CSV file
df.to_csv('data/churn_data.csv', index=False)

print("Synthetic churn data generated and saved to 'data/churn_data.csv'")

# --- EDA Section ---
import seaborn as sns
import matplotlib.pyplot as plt

print("Displaying EDA plots...")

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=df)
plt.title('Churn Distribution')


# 2. Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='contract', hue='churn', data=df)
plt.title('Churn by Contract Type')


# 3. Monthly Charges by Churn Status
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='monthly_charges', hue='churn', multiple='stack', kde=True)
plt.title('Monthly Charges Distribution by Churn Status')

plt.show()

# --- Preprocessing Section ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("\nPreprocessing data...")

# Separate features and target
X = df.drop('churn', axis=1)
y = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Create the preprocessing pipelines for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply the transformations
X_processed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print("Data preprocessed and split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- Neural Network Section ---
import tensorflow as tf
from tensorflow import keras

print("\nBuilding and training the neural network...")

# Define the model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=1)

# --- Model Evaluation Section ---
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("\n--- Model Evaluation ---")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()

# --- Save Model and Preprocessor ---
print("\nSaving model and preprocessor...")

# Save the Keras model
model.save('churn_model.h5')

# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor.joblib')

print("Model saved as 'churn_model.h5'")
print("Preprocessor saved as 'preprocessor.joblib'")
