import pandas as pd
import os

def clean_spectrum():
    """
    Cleans the consolidated IR spectrum CSV by handling missing data, and filter the targeted wavenumber range.
    
    Parameters:
    - input_file (str): Path to the consolidated CSV file. Refer to consolidate_spectrum.py for expected data structure.
    - output_file (str): Path where the cleaned CSV will be saved.
    - method (str): Method to handle missing data. Options: 'drop', 'interpolate', 'fill'.
    - fill_value (float): Value to fill missing data if method is 'fill'. Ignored otherwise.
    - wavenumber_min (float): Minimum wavenumber to include.
    - wavenumber_max (float): Maximum wavenumber to include.
    - verbose (bool): If True, prints detailed logs.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    # Check if input file exists
    print(f"Examine NULL from spectrums. Loading spectrum...")
    if not os.path.isfile(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' does not exist.")
        return None
    try:
        # Read the consolidated CSV
        if VERBOSE:
            print(f"Reading input file '{INPUT_FILE}'...")
        df = pd.read_csv(INPUT_FILE)

        # Ensure 'Wavenumber' column exists
        if 'Wavenumber' not in df.columns:
            print("Error: 'Wavenumber' column not found in the input file.")
            return None

        # Filter wavenumbers within the specified range
        if VERBOSE:
            print(f"Filtering wavenumbers between {WAVE_MIN} and {WAVE_MAX} cm⁻¹...")
        df = df[(df['Wavenumber'] >= WAVE_MIN) & (df['Wavenumber'] <= WAVE_MAX)]
        # Reset index after filtering
        df.reset_index(drop=True, inplace=True)

        # Display the number of missing values before cleaning
        missing_before = df.isnull().sum().sum()
        if VERBOSE:
            print(f"Total missing values before cleaning: {missing_before}")

        # Handle missing data based on the chosen method
        if METHOD == 'drop':
            if VERBOSE:
                print("Dropping rows with any missing values...")
            df_clean = df.dropna()
            action = "Dropped"
        elif METHOD == 'interpolate':
            if VERBOSE:
                print("Interpolating missing values...")
            # Interpolate along the rows (wavenumbers)
            df_clean = df.interpolate(method='linear', axis=0).dropna()
            action = "Interpolated"
        elif METHOD == 'fill':
            if FILL_VALUE is None:
                print("Error: 'fill_value' must be specified when using 'fill' method.")
                return None
            if verbose:
                print(f"Filling missing values with {FILL_VALUE}...")
            df_clean = df.fillna(FILL_VALUE)
            action = f"Filled with {FILL_VALUE}"
        else:
            print(f"Error: Unknown method '{METHOD}'. Choose from 'drop', 'interpolate', 'fill'.")
            return None

        # Display the number of missing values after cleaning
        missing_after = df_clean.isnull().sum().sum()
        if VERBOSE:
            print(f"Total missing values after cleaning: {missing_after}")
            
        # Save the cleaned DataFrame to CSV
        if VERBOSE:
            output_path = OUTPUT_FILE + '_Wavenumber_' + str(WAVE_MIN) + '-' + str(WAVE_MAX) + '.csv'
            print(f"Saving cleaned data to {output_path}...")
        df_clean.to_csv(output_path, index=False)
        if VERBOSE:
            print(f"Cleaned data saved successfully. ({action})")
        
        return df_clean

    except Exception as e:
        print(f"Error during cleaning process: {e}")
        return None


def remove_outliner(df_clean): 
    if VERBOSE:
        print(f"Removing outliner {OUTLINER} and saving to {OUTPUT_FILE}...")
    for i in range (len(OUTLINER)):
        df_clean_remove_outliner = df_clean.drop(OUTLINER[i], axis=1)
    # Save the processed DataFrame to CSV
    output_path = OUTPUT_FILE + '_Wavenumber_' + str(WAVE_MIN) + '-' + str(WAVE_MAX) + '.csv'
    df_clean_remove_outliner.to_csv(output_path, index=False)
    if VERBOSE:
        print(f"Outliner removed successfully.")
    
    return

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = './consolidated_spectrum_raw.csv'
    OUTPUT_FILE = './consolidated_and_cleaned'
    METHOD = 'interpolate' # Choose 'drop', 'interpolate', or 'fill'
    FILL_VALUE = None # Only used if 'METHOD' is 'fill'
    WAVE_MIN = 600 # Select the wavenumber range
    WAVE_MAX = 1400 # Select the wavenumber range
    VERBOSE = True
    OUTLINER = ['12.50'] # Optional: list the column(s) to be removed

    # Perform cleaning
    cleaned_df = clean_spectrum()
    
    # Optional: remove outliner
    remove_outliner(cleaned_df) # The first spectrum data found outliner at 12.5%