# main.py
# Customer Churn Prediction using a Neural Network
# This script builds a complete pipeline for a churn prediction task:
# 1. Generates a synthetic dataset for demonstration.
# 2. Performs Exploratory Data Analysis (EDA) with Seaborn.
# 3. Preprocesses the data (scaling, encoding).
# 4. Builds, trains, and evaluates a neural network with TensorFlow/Keras.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

# TensorFlow and Scikit-learn imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- 1. Data Generation ---
# In a real-world scenario, you would load your data here.
# For this example, we'll create a synthetic CSV file to make the script self-contained.
def create_churn_dataset(num_samples=5000, filename="data/churn_data.csv"):
    """Generates and saves a synthetic customer churn dataset."""
    if os.path.exists(filename):
        print(f"'{filename}' already exists. Loading existing data.")
        return pd.read_csv(filename)

    print(f"Generating a new synthetic dataset: '{filename}'")
    data = {
        'CustomerID': [f'CUST-{i:04d}' for i in range(num_samples)],
        'Gender': np.random.choice(['Male', 'Female'], num_samples, p=[0.5, 0.5]),
        'Age': np.random.randint(18, 70, num_samples),
        'TenureMonths': np.random.randint(1, 72, num_samples),
        'MonthlyCharges': np.random.uniform(20, 120, num_samples).round(2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples, p=[0.55, 0.25, 0.20]),
        'HasTechSupport': np.random.choice(['Yes', 'No'], num_samples, p=[0.4, 0.6]),
        'Churn': np.zeros(num_samples, dtype=int)
    }
    df = pd.DataFrame(data)

    # Introduce correlations to make the data more realistic
    # Churn is more likely with shorter tenure, higher charges, and month-to-month contracts
    churn_probability = (1 / (1 + np.exp(-(
        -0.08 * df['TenureMonths']
        + 0.03 * df['MonthlyCharges']
        - 0.05 * df['Age']
        + (df['Contract'] == 'Month-to-month') * 1.5
        - (df['Contract'] == 'Two year') * 1.5
        + (df['HasTechSupport'] == 'No') * 1.0
        - 3.5
    ))))
    
    df['Churn'] = (np.random.rand(num_samples) < churn_probability).astype(int)
    
    # Calculate TotalCharges based on Tenure and MonthlyCharges with some noise
    df['TotalCharges'] = (df['TenureMonths'] * df['MonthlyCharges'] * np.random.uniform(0.95, 1.05, num_samples)).round(2)
    # Handle cases where tenure is low, TotalCharges might be less than MonthlyCharges
    df.loc[df['TotalCharges'] < df['MonthlyCharges'], 'TotalCharges'] = df['MonthlyCharges']


    df.to_csv(filename, index=False)
    print(f"Dataset with {num_samples} samples saved to '{filename}'.")
    return df

# Generate or load the dataset
churn_df = create_churn_dataset()

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- Starting Exploratory Data Analysis ---")
print("\nDataset Head:")
print(churn_df.head())

print("\nDataset Info:")
churn_df.info()

print("\nChurn Distribution:")
print(churn_df['Churn'].value_counts(normalize=True))

# Set plot style
sns.set_style("whitegrid")

# Create a figure for EDA plots
plt.figure(figsize=(18, 12))
plt.suptitle("Exploratory Data Analysis: Customer Churn", fontsize=20)

# Plot 1: Churn distribution
plt.subplot(2, 3, 1)
sns.countplot(x='Churn', data=churn_df, palette='viridis')
plt.title('Churn Distribution (0 = No, 1 = Yes)')

# Plot 2: Churn by Contract type
plt.subplot(2, 3, 2)
sns.countplot(x='Contract', hue='Churn', data=churn_df, palette='magma')
plt.title('Churn by Contract Type')
plt.xticks(rotation=15)

# Plot 3: Churn by Tech Support
plt.subplot(2, 3, 3)
sns.countplot(x='HasTechSupport', hue='Churn', data=churn_df, palette='plasma')
plt.title('Churn by Tech Support Status')

# Plot 4: Monthly Charges Distribution by Churn
plt.subplot(2, 3, 4)
sns.histplot(data=churn_df, x='MonthlyCharges', hue='Churn', multiple='stack', kde=True, palette='cividis')
plt.title('Monthly Charges Distribution by Churn')

# Plot 5: Tenure Distribution by Churn
plt.subplot(2, 3, 5)
sns.histplot(data=churn_df, x='TenureMonths', hue='Churn', multiple='stack', kde=True, palette='inferno')
plt.title('Tenure (Months) Distribution by Churn')

# Plot 6: Correlation Heatmap
plt.subplot(2, 3, 6)
# Select only numeric columns for correlation matrix
numeric_cols = churn_df.select_dtypes(include=np.number)
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numeric Features')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print("\nDisplaying EDA plots...")
plt.show()


# --- 3. Data Preprocessing ---
print("\n--- Starting Data Preprocessing ---")

# Drop the customer ID as it's not a predictive feature
churn_df_processed = churn_df.drop('CustomerID', axis=1)

# Define categorical and numerical features
categorical_features = ['Gender', 'Contract', 'HasTechSupport']
numerical_features = ['Age', 'TenureMonths', 'MonthlyCharges', 'TotalCharges']

# Define the target variable
X = churn_df_processed.drop('Churn', axis=1)
y = churn_df_processed['Churn']

# Create preprocessing pipelines for numerical and categorical features
# Numerical features will be scaled.
# Categorical features will be one-hot encoded.
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Apply the preprocessing pipeline to the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# --- 4. Model Building with TensorFlow/Keras ---
print("\n--- Building the Neural Network Model ---")

# Get the number of input features after preprocessing
input_dim = X_train_processed.shape[1]

model = Sequential([
    # Input layer: Dense layer with 32 neurons, ReLU activation
    Dense(32, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2), # Dropout layer to prevent overfitting

    # Hidden layer: Dense layer with 16 neurons, ReLU activation
    Dense(16, activation='relu'),
    Dropout(0.2),

    # Output layer: Dense layer with 1 neuron, sigmoid activation for binary classification
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Suitable for binary classification
    metrics=['accuracy']
)

# Print model summary
model.summary()


# --- 5. Model Training ---
print("\n--- Training the Model ---")
# Use a portion of training data for validation during training
history = model.fit(
    X_train_processed,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2, # Use 20% of training data for validation
    verbose=1
)

# --- 6. Model Evaluation ---
print("\n--- Evaluating the Model ---")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Make predictions
y_pred_proba = model.predict(X_test_processed)
y_pred = (y_pred_proba > 0.5).astype(int).flatten() # Convert probabilities to binary predictions

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\n--- Project Complete ---")

# --- 7. Save Model and Preprocessor ---
print("\n--- Saving the model and preprocessor ---")

# Save the trained model to a file
model.save('churn_model.h5')
print("Model saved as 'churn_model.h5'")

# Save the preprocessor object
joblib.dump(preprocessor, 'preprocessor.joblib')
print("Preprocessor saved as 'preprocessor.joblib'")