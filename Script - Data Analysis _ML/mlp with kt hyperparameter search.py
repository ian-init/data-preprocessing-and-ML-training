import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
from tkinter import filedialog
import keras_tuner as kt
import random

def load_and_prepare_data(csv_path='./consolidated_spectrum_cleaned.csv'):
    """
    Loads and preprocesses the CSV data.
    The CSV is expected to have a 'Wavenumber' column and the remaining columns containing spectral data,
    with one column (after transposition) representing the target "Concentration".
    
    Returns:
        X_train_scaled (np.array): Scaled training features.
        X_test_scaled (np.array): Scaled testing features.
        y_train (np.array): Training labels.
        y_test (np.array): Testing labels.
        scaler (StandardScaler): Fitted scaler.
    """
    # Check if the file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File '{csv_path}' does not exist.")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    print("Data Types After Loading:")
    print(df.dtypes)
    
    # Drop rows with missing values if any
    if df.isnull().values.any():
        print("Warning: Missing values detected. Dropping rows with missing values.")
        df = df.dropna()
    
    # Transpose so that each row is a sample
    df_transposed = df.set_index('Wavenumber').transpose().reset_index()
    df_transposed = df_transposed.rename(columns={'index': 'Concentration'})
    
    print("\nData Types After Transposition:")
    print(df_transposed.dtypes)
    
    # Convert all columns to numeric (coerce errors to NaN)
    feature_columns = df_transposed.columns.drop('Concentration')
    df_transposed[feature_columns] = df_transposed[feature_columns].apply(pd.to_numeric, errors='coerce')
    df_transposed['Concentration'] = pd.to_numeric(df_transposed['Concentration'], errors='coerce')
    
    if df_transposed.isnull().values.any():
        print("Warning: Non-numeric values detected after conversion. Dropping such rows.")
        df_transposed = df_transposed.dropna()
    
    # Separate features and labels
    X = df_transposed.drop('Concentration', axis=1).values
    y = df_transposed['Concentration'].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\nData loaded and split. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler

def plot_history(history):
    """
    Plots the training and validation loss and MAE over epochs.
    
    Parameters:
    - history (kt.RandomSearch.callbacks.History): History object returned by model.fit().
    
    Returns:
    - None
    """    
    plt.figure(figsize=(14, 5))
    
    # Plot Loss (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and plots Actual vs Predicted values.
    
    Parameters:
    - model (kt.RandomSearch): Trained KerasTuner RandomSearch model.
    - X_test (np.array): Test features.
    - y_test (np.array): Test labels.
    
    Returns:
    - None
    """
    # Predict on the test set
    y_pred = model.predict(X_test).flatten()
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test Mean Squared Error (MSE): {mse:.4f}")
    print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Test R² Score: {r2:.4f}")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Concentration (%)')
    plt.ylabel('Predicted Concentration (%)')
    plt.title('Actual vs. Predicted Ilmenite Concentration')
    plt.grid(True)
    plt.show()

def build_model(hp):
    """
    Hypermodel function for KerasTuner.
    Uses the hp object to define the hyperparameter search space.
    The global variable `input_dim` (set in main) is used to define the input shape.
    """
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=(input_dim,)))
    
    # First dense layer with tunable units and dropout rate
    units1 = hp.Int('units1', min_value=64, max_value=256, step=32)
    model.add(Dense(units1, activation='relu'))
    dropout1 = hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(dropout1))
    
    # Optionally add a second dense layer
    if hp.Boolean('second_layer'):
        units2 = hp.Int('units2', min_value=32, max_value=128, step=16)
        model.add(Dense(units2, activation='relu'))
        dropout2 = hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout2))
    
    # Optionally add a third dense layer
    if hp.Boolean('third_layer'):
        units3 = hp.Int('units3', min_value=16, max_value=64, step=16)
        model.add(Dense(units3, activation='relu'))
        dropout3 = hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout3))
    
    # Output layer
    model.add(Dense(1, activation='linear'))
    
    # Tune the learning rate for the Adam optimizer
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def main():
    # Standardise random state in different models.
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Ask the user to select a CSV file
    csv_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    # Load and prepare the data
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(csv_path)
    
    # Set global variable for input dimension (used by build_model)
    global input_dim
    input_dim = X_train.shape[1]
    
    # Define early stopping to avoid overfitting during tuning and training
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )
    
    # Create a KerasTuner RandomSearch object to tune the hyperparameters
    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective='val_mae',
        max_trials=20,            # Number of hyperparameter combinations to try
        executions_per_trial=1,   # Number of models to build and fit for each trial
        directory='tuner_dir',    # Directory to save the tuner results
        project_name='ilmenite_tuning'
    )
    
    # Run the hyperparameter search
    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        epochs=1000,         # Fewer epochs during search for speed; can increase if desired
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Print a summary of the search results
    tuner.results_summary()
    
    # Retrieve the best model from the tuner
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # USe KerasTuner best result for final training
    print("Retraining the best model on the full training set...")
    history = best_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=1000,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Plot training history
    plot_history(history)
    
    # Evaluate the best model on the test set
    evaluate_model(best_model, X_test, y_test)
    
    # Save the best model and the scaler for future use
    model_save_path = './mlp_ilmenite_best_model.h5'
    scaler_save_path = './scaler.pkl'
    best_model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"Best model saved to '{model_save_path}'.")
    print(f"Scaler saved to '{scaler_save_path}'.")

if __name__ == "__main__":
    main()