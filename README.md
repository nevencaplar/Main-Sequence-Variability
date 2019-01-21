# Main-Sequence-Variability


The main file is the Main_Sequence_Variability.ipynb notebook. It has several examples
- overview of the autocorrelation functions and its connection with the parameters of the power spectrum density (high-frequency slope and timescale of the break)
- overview of the analytical results of the measured width of the Main Sequence, as a function of the parameters of the power spectrum density
- Figure 6 from the paper, showing the relation between the intrinsic Main Sequence and measured Main Sequence, in a toy model
- Analytical result for the Figure 7 from the manuscript, showing how we recover the parameters of the PSD from observations, in a toy model.

Dependencies are pandas, tqdm and DELCgen (https://github.com/samconnolly/DELightcurveSimulation).
 
### Screenshot of the notebook is below:
------------------------------------------------------------------------------------------

![Overview of the notebook](https://www.dropbox.com/s/zejyvj7xov6j771/MS_Variability.png?raw=1)
------------------------------------------------------------------------------------------
### Other interesting files are:

- MS_Variability.py - module with definitions for Main_Sequence_Variability.ipynb

- ACFTableLargeNov10.csv - tabulated auto-correlation functions, used in the analytical analysis

- CreateACFTableFlatten.nb - code used to create tabulated auto-correlation functions
