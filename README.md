# Customer Churn Prediction

## AI/ML Engineer Project: End-to-End Customer Churn Prediction System

This project implements a comprehensive machine learning pipeline for predicting customer churn, leveraging a neural network model. It covers the entire lifecycle from synthetic data generation and exploratory data analysis (EDA) to advanced data preprocessing, model training, rigorous evaluation, and deployment via an interactive web dashboard for real-time predictions. This system is designed to help businesses proactively identify at-risk customers and implement retention strategies.

## Key Features

*   **Neural Network Model**: Developed and trained a robust neural network using TensorFlow/Keras for accurate binary classification of customer churn.
*   **Synthetic Data Generation**: Includes a module for generating realistic synthetic customer datasets, enabling reproducible research and demonstration.
*   **Comprehensive Data Pipeline**: Features a `scikit-learn` pipeline for efficient data preprocessing, including numerical feature scaling and one-hot encoding for categorical variables.
*   **Model Evaluation**: Provides in-depth model performance analysis, including accuracy, classification reports, confusion matrices, and visualization of training history to monitor for overfitting.
*   **Interactive Streamlit Dashboard**: Deploys the trained model as a user-friendly web application, allowing stakeholders to input customer data and receive instant churn predictions.
*   **Model & Preprocessor Persistence**: Ensures reusability and seamless deployment by saving the trained Keras model (`.h5`) and the `scikit-learn` preprocessor (`.joblib`).

## Technical Stack

*   **Programming Language**: Python
*   **Machine Learning Frameworks**: TensorFlow, Keras, Scikit-learn
*   **Data Manipulation & Analysis**: Pandas, NumPy, Matplotlib, Seaborn
*   **Web Framework**: Streamlit
*   **Model Persistence**: Joblib
*   **Environment Management**: Virtual Environments

## Project Structure

```
. 
├── data/ 
│   └── churn_data.csv        # Synthetic customer data
├── dashboard.py              # Streamlit web application for interactive predictions
├── generate_data.py          # Script for data generation, EDA, preprocessing, training, and saving the model
├── main.py                   # Orchestrates the full ML pipeline
├── requirements.txt          # Python dependencies
├── churn_model.h5            # Saved, trained Keras model
└── preprocessor.joblib       # Saved scikit-learn preprocessor
```

## Setup and Local Execution

Follow these instructions to set up and run the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/VamshiSunnam/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Full ML Pipeline

Execute `main.py` to generate data (if not present), perform EDA, preprocess, train, and evaluate the neural network model. This will also save the trained model and preprocessor.

```bash
python main.py
```

### 5. Launch the Interactive Dashboard

Once the model is trained and saved, launch the Streamlit dashboard to interact with the churn prediction system:

```bash
streamlit run dashboard.py
```

Open your web browser and navigate to the local URL provided by Streamlit.

## Learnings and Future Enhancements

This project provided deep insights into building end-to-end ML solutions, particularly in neural network design, hyperparameter tuning, and deploying interactive applications. Future work could involve integrating real-world customer data, exploring advanced deep learning architectures, implementing A/B testing for model versions, and deploying the solution to a cloud-based MLOps platform for continuous integration and delivery.
