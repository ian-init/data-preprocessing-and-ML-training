import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def polynomial_regression(
    input_file='./consolidated_spectrum_cleaned.csv',
    degree=2
    ):

    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return None

    else:
        # Load the data
        df = pd.read_csv(input_file, header=None)
        
        # Define dependent (y) and independent variables (X)
        X = df.iloc[1:, 1:].to_numpy()  # All rows except the first, excluding the first column
        y = df.iloc[0, 1:].to_numpy()  # First row, excluding the first element
        print(f"No of features: {len(X)}")
        print(f"No of targerts: {len(y)}")
        X = X.T # Transpose X so that each column corresponds to one observation

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Apply Polynomial Features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Train the model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_poly)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print results
        print(f"Model Coefficients:, {model.coef_}")
        print(f"Model Intercept: {model.intercept_}")
        print(f"Test Mean Squared Error (MSE): {mse:.4f}")
        print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Test R² Score: {r2:.4f}")

    return model.coef_, model.intercept_, mse, mae, r2

if __name__ == "__main__":
    # Test the polynomial regression with degree 2
    polynomial_regression(degree=2)