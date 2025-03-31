import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate_single_model(model, X_train, y_train, X_test, y_test):
    """
    Fits the model, predicts the targets, and returns the intercept, evaluation metrics, and predictions.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model.intercept_, mse, mae, r2, y_pred

def run_model_by_alpha(model_class, alphas, X_train, y_train, X_test, y_test,
                         fit_intercept, max_iter, tol, **extra_params):
    """
    Iterates over a list of alpha values for the given model_class, collects evaluation metrics and predictions for each run.
    """
    intercept_list = []
    mse_list = []
    mae_list = []
    r2_list = []
    y_pred_list = []
    for alpha in alphas:
        model = model_class(alpha=alpha, fit_intercept=fit_intercept,
                            max_iter=max_iter, tol=tol, **extra_params)
        intercept, mse, mae, r2, y_pred = evaluate_single_model(model, X_train, y_train, X_test, y_test)
        intercept_list.append(intercept)
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)
        y_pred_list.append(y_pred)
    return intercept_list, mse_list, mae_list, r2_list, y_pred_list

def plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted"):
    """
    Plots the actual values versus predicted values. A red dashed line indicates the perfect prediction scenario.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
    
    # Create a line for perfect predictions
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.show()

def sklearn_regression(model_para, ALPHA, FIT_INTERCEPT, MAX_ITER, TOL, SOLVER, WARM_START):
    """
    Assumption: Source spectrum is already baseline corrected and normalised.
    See and use baseline_and_normalise.py for methdology and execution
    """

    # Load spectrum
    print(f"Execute regession. Loading spectrum...")
    input_file = filedialog.askopenfilename(title="Select a File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    file_name = os.path.basename(input_file) # Extract file name
    extract_wavelength = r'\d{1,3}-\d{1,4}' # Regex pattern to extract wavelength

    # Attempt extraction of wavelength from the filename
    match = re.search(extract_wavelength, file_name)
    source_wavenumber = match.group() if match else "Unknown"

    print(f"Model selected: {model_para}")
    print(f"Source wavenumber range: {source_wavenumber}")

    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    print("Start data extraction")
    df = pd.read_csv(input_file, header=None)
    X = df.iloc[1:, 1:].to_numpy()
    y = df.iloc[0, 1:].to_numpy()
    print(f"No of features: {len(X)}")
    print(f"No of targets: {len(y)}")
    X = X.T

    print("Splitting train/test set")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For Ridge and Lasso, create a list of alpha values
    alpha_range = list(range(1, int(ALPHA)))

    print("Spinning up the model")
    if model_para == 'Linear':
        model = LinearRegression(fit_intercept=FIT_INTERCEPT)
        intercept, mse, mae, r2, y_pred = evaluate_single_model(model, X_train, y_train, X_test, y_test)
        print(f"Model Intercept: {intercept}")
        print(f"Test Mean Squared Error (MSE): {mse:.4f}")
        print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Test R² Score: {r2:.4f}")
        # Plot y_test vs. y_pred for Linear regression
        plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted (Linear)")

    elif model_para == 'Ridge':
        intercept_list, mse_list, mae_list, r2_list, y_pred_list = run_model_by_alpha(
            Ridge, alpha_range, X_train, y_train, X_test, y_test,
            FIT_INTERCEPT, MAX_ITER, TOL, solver=SOLVER
        )
        print(f"The minimum Mean Squared Error (MSE) value achieved: {min(mse_list)}")
        print(f"The minimum Test Mean Absolute Error (MAE) value achieved: {min(mae_list)}")
        print(f"The maximum R² Score achieved: {max(r2_list)}")
        
        # Identify the best model (based on the lowest MSE) and plot its predictions.
        best_index = mse_list.index(min(mse_list))
        best_alpha = alpha_range[best_index]
        best_y_pred = y_pred_list[best_index]
        print(f"Best model at alpha={best_alpha} with MSE={mse_list[best_index]:.4f}")
        plot_actual_vs_predicted(y_test, best_y_pred, title=f"Actual vs Predicted (Ridge, alpha={best_alpha})")

    elif model_para == 'Lasso':
        intercept_list, mse_list, mae_list, r2_list, y_pred_list = run_model_by_alpha(
            Lasso, alpha_range, X_train, y_train, X_test, y_test,
            FIT_INTERCEPT, MAX_ITER, TOL, warm_start=WARM_START
        )
        print(f"The minimum Mean Squared Error (MSE) value achieved: {min(mse_list)}")
        print(f"The minimum Test Mean Absolute Error (MAE) value achieved: {min(mae_list)}")
        print(f"The maximum R² Score achieved: {max(r2_list)}")
        
        # Identify the best model (based on the lowest MSE) and plot its predictions.
        best_index = mse_list.index(min(mse_list))
        best_alpha = alpha_range[best_index]
        best_y_pred = y_pred_list[best_index]
        print(f"Best model at alpha={best_alpha} with MSE={mse_list[best_index]:.4f}")
        plot_actual_vs_predicted(y_test, best_y_pred, title=f"Actual vs Predicted (Lasso, alpha={best_alpha})")
    else:
        print("Incorrect model selected")
        return

    # For Ridge and Lasso, also visualize how the metrics change with alpha
    if model_para in ['Ridge', 'Lasso']:
        print("Visualising result")
        plt.figure(figsize=(15, 5))
        plt.title(f"Model: {model_para} regression, Wavelength range: {source_wavenumber}\n")

        # Plot MSE vs. Alpha
        plt.subplot(1, 4, 1)
        plt.plot(alpha_range, mse_list, marker='o', linestyle='-')
        plt.title("MSE vs. Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Mean Squared Error")

        # Plot MAE vs. Alpha
        plt.subplot(1, 4, 2)
        plt.plot(alpha_range, mae_list, marker='o', color='green', linestyle='-')
        plt.title("MAE vs. Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Mean Absolute Error")

        # Plot R² vs. Alpha
        plt.subplot(1, 4, 3)
        plt.plot(alpha_range, r2_list, marker='o', color='red', linestyle='-')
        plt.title("R² vs. Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("R² Score")

        # Plot model intercept vs. Alpha
        plt.subplot(1, 4, 4)
        plt.plot(alpha_range, intercept_list, marker='o', color='purple', linestyle='-')
        plt.title("Model Intercept vs. Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Model Intercept")

        plt.tight_layout()
        plt.show()
        plt.close()

    print("Modelling completed")

if __name__ == "__main__":  
    # Configuration
    MODEL = 'Ridge'  # Choose from 'Linear', 'Ridge' or 'Lasso'

    # Common parameters
    FIT_INTERCEPT = True  # default value True
    # Common for Ridge and Lasso: maximum alpha value for iteration
    ALPHA = 20
    MAX_ITER = 1000      # default value: Ridge -> None, Lasso -> 1000
    TOL = 0.001        # default value: 0.001
    # Ridge-specific parameter
    SOLVER = 'auto'    # Options: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'
    # Lasso-specific parameter
    WARM_START = True

    sklearn_regression(MODEL, ALPHA, FIT_INTERCEPT, MAX_ITER, TOL, SOLVER, WARM_START)