<p>The repository aims to build a data pipeline for chemical analysis ML model. The modules cleans, baseline, and normalized spectrometry readings. It also find the best paremeters for regression and MLP models (by singal-to-noise and Mean Squared Error (MSE)) and train the model.</p>

<h4>Files hierarchy</h4>
<ol>
    <li>
        <p><b>File: </b>consolidated_raw.py</p>
        <p><b>Function: </b>Combine multiple spectrums</p>
        <p><b>Input: </b>CSVs of spectrum reading / Data structure: Header: None, First column: wavenumber, Second column: IR reading</p>
        <p><b>Output: </b>Single CSV file: consolidated_spectrum_raw.csv / Data structure: Header: Yes, First column: wavenumber, From second column: IR readings</p> 
    </li>
    <li>
        <p><b>File: </b>plot_spectrum.py</p>
        <p><b>Function: </b>Visualise spectrums to identify outliner</p>
        <p><b>Input: </b>CSV. Refer to consolidated_raw.py for expected data structure</p>
        <p><b>Output: </b>Grpah on terminal</p>
    </li>
    <li>
        <p><b>File: </b>clean_data.py</p>
        <p><b>Function: </b>i)Remove NaN and filter fingerprint region wavenumber range, and ii)Remove outliner(s)</p>
        <p><b>Input: </b>CSV. Refer to consolidated_raw.py for expected data structure</p>
        <p><b>Output: </b>consolidated_spectrum_cleaned_wavlength_<i>{wavenumber selected}</i></p> 
    </li>
	<li>
		<p><b>File: </b>baseline_and_normalise.py</p>
		<p><b>Function: </b>i)Baseline, and ii) Normalise IR reading</p>
        <p>The script used both PeakUtils and ALS baseline method. User manully select between Standard Scalar and MinMax Scalar. User can optionally vaires the model parameters (or use the default). The script then iterate throight the parameters range and find the best signal-to-noise models.</p>  
		<p><b>Input: </b>CSV. Refer to consolidated_raw.py for expected data structure</p>
		<p><b>Output: </b>Peakutils_<i>{normalised method}</i>_Wavenumber_<i>{wavenumber range}</i>.csv" and "ALS_<i>{normalised method}</i>_Wavenumber_<i>{wavenumber range}</i>.csv"</p> 
	</li>
	<li>
		<p><b>File: </b>mlp.py</p>
		<p><b>Function: </b>Train cleand, baselined, and normalised spectrums with MLP</p>
		<p><b>Input: </b>CSV. Refer to baseline_and_normalise.py for expected data structure</p>
		<p><b>Output:</b> i)Grpah on terminal, ii) Save trained model and scalar as h5 and pkl file</p>
	</li>
    <li>
        <p><b>File: </b>sklearn_lenearregression.py</p>
		<p><b>Function: </b>Train cleand, baselined, and normalised spectrums with Linear, Ridge, or Lasso regression</p>
        <p>User manully select the regression method. User can optionally varies the model parameters (or use the default). The script then iterate throight the parameters range and find the best Mean Squared Error (MSE) model.</p>
        <p><b>Input: </b>CSV. Refer to baseline_and_normalise.py for expected data structure</p>
        <p><b>Output: </b>Grpahs on terminal</p>        
</ol>
<hr>
<p>Output example</p>
<img src="https://raw.githubusercontent.com/ian-init/data-preprocessing-and-ML-training/refs/heads/main/trail%20sample/MLP_ALS_MinMaxScaler.png">
<img src="https://raw.githubusercontent.com/ian-init/data-preprocessing-and-ML-training/refs/heads/main/trail%20sample/ridge%20regression_wavelength_600-1400_alpha200_max_iter_none.png">
<img src="https://raw.githubusercontent.com/ian-init/data-preprocessing-and-ML-training/refs/heads/main/trail%20sample/MLP_ALS_StandardScaler_pred.png">