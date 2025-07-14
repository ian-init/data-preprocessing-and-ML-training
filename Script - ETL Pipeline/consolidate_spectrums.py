import pandas as pd
import glob
import os
import re

def combine_spectrum():
    """
    Combines multiple IR data CSV files into a single consolidated CSV file.

    Assumption:
    - Each spectrum is in an indvidual CSV file.
    - Each CSV file have 2 columns only: the 1st one is the wavenumber, and the 2nd one is the reading.
    - the columns do not have header row  

    Parameters:
    - data_folder (str): Path to the folder containing the CSV files.
    - output_file (str): Path where the consolidated CSV will be saved.

    Returns:
    - pd.DataFrame: The consolidated DataFrame.
    """
    
    print(f"Consolidate raw IR spectrum from folder '{DATA_FOLDER}'. Loading raw files...")
    # Define the pattern to match filenames like Il_8-00.csv or Il_8-00.CSV
    file_pattern = os.path.join(DATA_FOLDER, 'Il_*.[cC][sS][vV]')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No files found matching the pattern.")
        return None
    
    # Initialize a list to hold individual spectrum as DataFrames
    dfs = []
    # Regular expression to extract concentration from filename
    concentration_pattern = re.compile(r'Il_(\d+)-(\d+)\.[cC][sS][vV]$') # Matches patterns like Il_8-00.csv or Il_8-00.CSV
    
    for file in files:
        basename = os.path.basename(file)
        match = concentration_pattern.match(basename)
        if match:
            integer_part, decimal_part = match.groups()
            concentration = float(f"{integer_part}.{decimal_part}")
        else:
            print(f"Filename {basename} does not match the expected pattern. Skipping.")
            continue
        
        try:
            # Read the CSV file without headers
            df = pd.read_csv(file, header=None, delimiter=None)
            if df.shape[1] < 2:
                print(f"File {basename} does not have at least two columns. Skipping.")
                continue
            # Rename columns to 'Wavenumber' and the concentration value
            df = df.iloc[:, :2]  # Ensure only first two columns are used
            df.columns = ['Wavenumber', f"{concentration:.2f}"]
            # Filter wavenumbers within the specified range
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {basename}: {e}")
            continue
    
    # if the raw CSV are not compatabile to the format lised 
    if not dfs:
        print("No valid dataframes to merge.")
        return None
    
    # Merge all DataFrames on 'Wavenumber' using outer join to include all wavenumbers
    print(f"Combining files")
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Wavenumber', how='outer')
    
    # Sort the merged DataFrame by 'Wavenumber'
    merged_df.sort_values('Wavenumber', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # Optionally, sort the concentration columns in ascending order
    concentration_columns = sorted([col for col in merged_df.columns if col != 'Wavenumber'], key=lambda x: float(x))
    merged_df = merged_df[['Wavenumber'] + concentration_columns]
    
    # Save the consolidated DataFrame to CSV
    merged_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Consolidated CSV saved to {OUTPUT_FILE}")
    return merged_df

if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = './Data'
    OUTPUT_FILE = './consolidated_spectrum_raw.csv'

    combined_spectrum = combine_spectrum()