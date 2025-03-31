import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import filedialog


"""
Visualise spectrun in graph.

Parameters:
- input_file (str): Path to the consolidated CSV file. Refer to consolidate_spectrums.py for expected data structure.
- output: Graph on terminal.

Returns:
- None
"""

# Load the spectrum file
input_file = filedialog.askopenfilename(title="Select a File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
df = pd.read_csv(input_file)  # Load without headers

# Extract "Wavenumber" for the x-axis and all other columns for the y-axis
x = df['Wavenumber']
columns_to_plot = df.columns[1:]  # Exclude the "Wavenumber" column

# Plot all series in the dataset
plt.figure(figsize=(14, 14))

for column in columns_to_plot:
    plt.plot(x, df[column], label=column)

# Add labels, title, legend, and grid
plt.xlabel('Wavenumber')
plt.ylabel('Transmittance')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Position the legend outside the plot
plt.grid(True)
plt.tight_layout()
plt.show()