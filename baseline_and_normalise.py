import os
import re
from tkinter import filedialog
import numpy as np
import pandas as pd
import peakutils
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

"""
Baseline corect spectrum data using both PeakUtils and Asymmetric Least Squares (ALS) method, and find the model with best Singal-to-Noise.
Normslise baselined model using either StandardScalar() or MinMaxScalar()

Parameters:
- input_file (str): Path to the consolidated CSV file. Refer to consolidate_spectrum.py for expected data structure.
- output_file (str): "Peakutils_{NOR_METHOD}_Wavenumber_{source_wavenumber}.csv" and "ALS_{NOR_METHOD}_Wavenumber_{source_wavenumber}.csv"

# Baseline
- LAM_VALUES (list): configurate the smoothness of baseline model
- P_VALUES (list): Asymmetry Parameter to defines the weighting between fitting above and below
- DEG (int): PeakUtils polynomial degree
- NITER (int): ALS number of iterations

# Normalisation
- NOR_METHOD (str): Normalisation method

Returns:
- pd.DataFrame: The baselined and normailised DataFrame.
"""

def load_spectrum():
    # Load dataset
    input_file = filedialog.askopenfilename(title="Select a File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    # Check if the file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File '{input_file}' does not exist.")

    print(f"Loading spectrum file ...")
    df = pd.read_csv(input_file)

    # From file name, Look for wavelength range
    file_name = os.path.basename(input_file) # Extract file name
    extract_wavelength = r'\d{1,3}-\d{1,4}' # Regex pattern to extract wavelength
    match = re.search(extract_wavelength, file_name)
    global source_wavenumber
    source_wavenumber = match.group() if match else "whole spectrum"
    print(f"Source wavenumber range: {source_wavenumber}")

    return df

# PeakUtils baseline correction
def baseline_peakutils(y, MAX_INT, deg):
    return peakutils.baseline(y, deg=deg)

# ALS baseline correction
def baseline_als(y, lam, p, NITER):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(NITER):
        W = diags(w, 0)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

#  calculate Singal to Noise ratio
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Compare parameters for best model
def finding_best_model(LAM_VALUES, P_VALUES, NITER, MAX_INT, DEG):
    print("Initiating baseline model selection")
    best_snr_als, best_rmse_als, best_params_als = -np.inf, np.inf, None
    best_snr_peakutils, best_rmse_peakutils, best_params_peakutils = -np.inf, np.inf, None
    # Extract wavenumbers for CSV data structure
    wavenumbers = df.iloc[:, 0]

    # PeakUtils Baseline Correction
    for deg in range (DEG + 1):
        corrected_peakutils = df.iloc[:, 1:].apply(lambda col: col.values - baseline_peakutils(col.values, MAX_INT, deg), axis=0)
        rmse_peakutils = root_mean_squared_error(df.iloc[:, 1:], corrected_peakutils)
        snr_peakutils = calculate_snr(corrected_peakutils.values, df.iloc[:, 1:].values - corrected_peakutils.values)
        if snr_peakutils > best_snr_peakutils and rmse_peakutils < best_rmse_peakutils:
            best_snr_peakutils, best_rmse_peakutils, best_deg_peakutils = snr_peakutils, rmse_peakutils, deg
            # Reconstruct dataframes
            best_peakutils_df = pd.concat([wavenumbers, corrected_peakutils], axis=1)
    print(f"Best PeakUtils - Polynomial degree: {best_deg_peakutils}, RMSE: {best_rmse_peakutils:.4f}, SNR: {best_snr_peakutils:.4f} dB")
    
    # ALS Baseline Correction
    for lam in LAM_VALUES:
        for p in P_VALUES:
            corrected_als = df.iloc[:, 1:].apply(lambda col: col.values - baseline_als(col.values, lam, p, NITER), axis=0)
            rmse_als = root_mean_squared_error(df.iloc[:, 1:], corrected_als)
            snr_als = calculate_snr(corrected_als.values, df.iloc[:, 1:].values - corrected_als.values)
            
            if snr_als > best_snr_als and rmse_als < best_rmse_als:
                best_snr_als, best_rmse_als, best_params_als = snr_als, rmse_als, (lam, p)
                # Reconstruct dataframes
                best_als_df = pd.concat([wavenumbers, corrected_als], axis=1)
    print(f"Best ALS - Lambda: {best_params_als[0]}, p: {best_params_als[1]}, RMSE: {best_rmse_als:.4f}, SNR: {best_snr_als:.4f} dB")
       
    return best_peakutils_df, best_als_df

# Function to normalize data
def normalize_data(best_model):
    if NOR_METHOD == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif NOR_METHOD == "StandardScaler":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Choose 'minmax' or 'standard'.")
    
    print(f"Initiating normalization with {NOR_METHOD} method")    
    peakUtils_normalized_values = scaler.fit_transform(best_model[0].iloc[:, 1:])
    peakUtils_normalised_df = pd.concat([best_model[0].iloc[:, [0]], pd.DataFrame(peakUtils_normalized_values, columns=best_model[0].columns[1:])], axis=1)

    als_normalized_values = scaler.fit_transform(best_model[1].iloc[:, 1:])
    als_normalised_df = pd.concat([best_model[1].iloc[:, [0]], pd.DataFrame(als_normalized_values, columns=best_model[1].columns[1:])], axis=1)

    # Export the models for ML training
    print("Exporting the completed DataFrame") 
    peakUtils_normalised_df.to_csv(f"PeakUtils_{NOR_METHOD}_Wavenumber_{source_wavenumber}.csv", index=False)
    als_normalised_df.to_csv(f"ALS_{NOR_METHOD}_Wavenumber_{source_wavenumber}.csv", index=False)

    return peakUtils_normalised_df, als_normalised_df

if __name__ == "__main__":
    # Configuration
    # ALS
    LAM_VALUES = [1e3, 1e4, 1e5, 1e6, 1e7] # controls the smoothnes
    P_VALUES = [0.001, 0.01, 0.1, 0.5, 0.9] # Asymmetry Parameter, defines the weighting between fitting above and below
    NITER = 10 # Number of Iterations
    # PeakUtils
    MAX_INT = 100 # (default: 100) Maximum number of iterations to perform.
    DEG = 3 # (default: 3) Polynomial degree
    # Normalization
    NOR_METHOD = "StandardScaler" # Choose between "MinMaxScaler" and "StandardScaler"

    # Program start with spectrum loading
    df = load_spectrum()
    # Baseline correction with Peakutils and ALS models
    best_model = finding_best_model(LAM_VALUES, P_VALUES, NITER, MAX_INT, DEG) # Function return 2 DataFrame, 1st one is Peakutils model, 2nd one is ALS model"
    # Normalization data
    normalize_df = normalize_data(best_model)