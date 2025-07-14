
import pandas as pd
import numpy as np
import os
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
from tkinter import filedialog


def load_and_prepare_data():
    """
    Loads the baselined and normalised CSV file and prepares features and labels for the MLP model.
    
    Parameters:
    - input_file (str): Path to the baselined and normalised CSV file. Refer to baseline_and_normalise.py for expected data structure.
    
    Returns:
    - X_train (np.array): Scaled training features.
    - X_test (np.array): Scaled testing features.
    - y_train (np.array): Training labels.
    - y_test (np.array): Testing labels.
    - scaler (object): Scaler method.
    - source_wavenumber: Wavenumber range used.

    """
    # Ask the user to select a CSV file
    input_file = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))

    # Check if the file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File '{input_file}' does not exist.")

    print(f"Loading spectrum file ...")
    file_name = os.path.basename(input_file) # Extract file name

    # From file name, Look for baseline method used.
    extract_baseline = r'^(ALS|PeakUtils)' # The fileanme start with the baseline method.
    match_base = re.search(extract_baseline, file_name)
    source_baseline = match_base.group() if match_base else "Unknown"
    print(f"Source baseline method: {source_baseline}")
    
    # From file name, Look for scalar method used.
    extract_normalisation = r'(?<=_)(StandardScaler|MinMaxScaler)(?=[_.])' # The fileanme denoted the method between two underscores.
    match_nor = re.search(extract_normalisation, file_name)
    if match_nor:
        scaler_match = match_nor.group()
        if scaler_match == "StandardScaler":
            source_scalar = StandardScaler()
        elif scaler_match == "MinMaxScaler":
            source_scalar = MinMaxScaler()
        else:
            print("Error: Normalisation method not found in the input file.")
            return None
        print(f"Source normalisation method: {source_scalar}")
    
    # From file name, Look for wavelength range
    extract_wavelength = r'\d{1,3}-\d{1,4}' # Regex pattern to extract wavelength
    match_wave = re.search(extract_wavelength, file_name)
    source_wavenumber = match_wave.group() if match_wave else "whole spectrum"
    print(f"Source wavenumber range: {source_wavenumber}")

    # Load the cleaned data without specifying dtypes
    df = pd.read_csv(input_file)
       
    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Missing values detected. Proceeding to drop rows with missing values.")
        df = df.dropna()
    
    # Transpose the DataFrame so that each row represents a sample
    df_transposed = df.set_index('Wavenumber').transpose().reset_index()
    df_transposed = df_transposed.rename(columns={'index': 'Concentration'})
    
    # Convert all feature columns to float
    feature_columns = df_transposed.columns.drop('Concentration')
    df_transposed[feature_columns] = df_transposed[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    # Convert 'Concentration' to float
    df_transposed['Concentration'] = pd.to_numeric(df_transposed['Concentration'], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    if df_transposed.isnull().values.any():
        print("Warning: Non-numeric values detected after conversion. Dropping such rows.")
        df_transposed = df_transposed.dropna()
    
    # Separate features and labels
    X = df_transposed.drop('Concentration', axis=1).values
    y = df_transposed['Concentration'].values
       
    # Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train /split completed")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, source_baseline, source_scalar, source_wavenumber

def build_mlp_model(input_dim):
    """
    Builds an MLP model using TensorFlow Keras.
    
    Parameters:
    - input_dim (int): Number of input features.
    
    Returns:
    - model (tf.keras.Model): Compiled MLP model.
    """
    model = Sequential()
    
    # Input Layer
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    
    # Hidden Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(1, activation='linear'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("MLP model built and compiled.")
    return model

def plot_history(history, source_baseline, source_scalar, source_wavenumber):
    """
    Plots the training and validation loss and MAE over epochs.
    
    Parameters:
    - history (tf.keras.callbacks.History): History object returned by model.fit().
    
    Returns:
    - None
    """
    # Plot Loss
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title(f'MLP Model Loss\nMethod: {source_baseline}, {source_scalar}\nWavenumber: {source_wavenumber}')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'Model Mean Absolute Error\nMethod: {source_baseline}, {source_scalar}\nWavenumber: {source_wavenumber}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.show()

def evaluate_model(model, X_test, y_test, source_baseline, source_scalar, source_wavenumber):
    """
    Evaluates the trained model on the test set and plots Actual vs Predicted.
    
    Parameters:
    - model (tf.keras.Model): Trained Keras model.
    - X_test (np.array): Test features.
    - y_test (np.array): Test labels.
    
    Returns:
    - None
    """
    # Predict on test set
    y_pred = model.predict(X_test).flatten()
    
    # Calculate evaluation metrics using Scikit-Learn
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
    plt.title(f'Actual vs. Predicted Ilmenite Concentration\nMethod: {source_baseline}, {source_scalar}\nWavenumber: {source_wavenumber}')
    plt.grid(True)
    plt.show()

def main():


    # Standardise random state in different models.
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Get train/test spilt
    X_train, X_test, y_train, y_test, source_baseline, source_scalar, source_wavenumber = load_and_prepare_data()
    
    # Build the MLP model
    input_dim = X_train.shape[1]
    model = build_mlp_model(input_dim)
    
    # Set up Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=1000,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )
        
    # Plot training history
    plot_history(history, source_baseline, source_scalar, source_wavenumber)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, source_baseline, source_scalar, source_wavenumber)
    
    # Save the trained model
    model_save_path = './mlp_ilmenite_model_' + source_baseline + '_' + str(source_scalar) + '_' + 'wavenumber_' + source_wavenumber + '.h5'
    model.save(model_save_path)
    print(f"Trained model saved to '{model_save_path}'.")
    
    # Save the scaler
    scaler_save_path = './scaler_' + source_baseline + '_' + str(source_scalar) + '_' + 'wavenumber_' + source_wavenumber + '.pkl'
    joblib.dump(source_scalar, scaler_save_path)
    print(f"Scaler saved to '{scaler_save_path}'.")
    
    # Example: Loading and using the model for prediction
    # loaded_model = tf.keras.models.load_model(model_save_path)
    # loaded_scaler = joblib.load(scaler_save_path)
    # new_sample = np.array([/* your transmittance values here */])
    # new_sample_scaled = loaded_scaler.transform(new_sample.reshape(1, -1))
    # predicted_concentration = loaded_model.predict(new_sample_scaled)
    # print(f"Predicted Ilmenite Concentration: {predicted_concentration[0][0]:.2f}%")

if __name__ == "__main__":
    main()