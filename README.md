The files on this repository are directly related to our paper: Cyclic Sulfone Ring
Remodling Enables Molecular Shape Diversity-Oriented Synthesis of Privileged
Biaryl and Oligoaryl Motifs. *Pending DOI*
*PLEASE* note that these codes do not contain methods for extracting descriptors, this
only houses python codes that were used to develop our classification model and generate 
graphs for normalized prinicple moments of Inertia (NMPIs). All descriptors were 
collected from detailed Gaussian 16 log files using bash scripts and manual labor. 

# REQUIREMENTS
All scripts were performed on Python 3.10.12 on Microsoft Windows
numpy==1.26.4
pandas==2.3.3
rdkit==2025.9.4
statsmodels=0.14.6
scikit-learn==1.3.2
#Has not been test on additional softwares or operating systems

# INSTALLATION 
All python files can be downloaded as a ZIP file and extracted on personal computer or HPC.
Ready to use after extraction of the files (less then 1 minute if all dependencies were 
installed previously).

# DEMO 
All codes can be run using any environment that can communicate with python. Our codes were
done on a personal computer with 16gbs of RAM and 4 processors. User can access a windows
terminal via CMD and run the codes as "python -m pythonfile". All scripts ran for less than 
a minute.

# Instructions for using codes
## NMPI Graphs
Codes associated for the <ins>NMPI graphs</ins> can run independently and **require** all files
be directly located in the same directory/folder. 
> Place the following files in one directory/folder:
> biphenyl_positions.csv
> all-figures_nprs-removedNP.csv
> chembel_nprs_last.csv
> cleaned-smiles.csv
> CalculateNormalizedPrincipalMomentsInertia.py
> ClassificationOfBiphenylSystems.py
> split-poly.py
> PlotNMPI.py
User can run "python -m pythonfile" on any of the files with the python (.py) extension listed
above.
### How to use
We have provided a template csv file that can be used as a test run. First the user must run the 
CalculateNormalizedPrincipleMomentsInertia.py script in python to collect a new csv file with two
columns that contains the NMPIs in the x and y dimension. This data may be copied over to the
cleaned-smiles.csv file and saved as either all-figures_nprs-removedNP.csv or chembel_nprs_last.csv.
**We recommend** the former for a clearer graph. You may run the PlotNMPI.py script to generate a
graph similar to the one shown in **Figure 1.C** of the main text. 
For classification of biphenyl systems, users may run ClassificationOfBiphenylSystem.py. This will
classify the substitution pattern in all the SMILES in chembel_nprs_last.csv and output a file
called biphenyl_categories_with_positions.csv. Users can modify this output file to appear like the 
one demonstrated in biphenyl_positions.csv. Finally users can separate each pattern, or combination
of patterns, into separate csv files using spilt-poly.py. This was used to tabulate substitution 
patterns in CHEMBL database. Examples of output files are located in the *Output files* directory.

## Model development
User can use our model_development.py script on the dataset.xlsx file included in datasets directory.
To run code, users will need to modify the lines in 236, 271 and 272 (last two lines) of the
model_development.py script to change the paths. Currently these lines are directed towards an 
external user path but can be modified to fit specific user. Outcome of results can be found in a
generated output file "output_18_04_2025.xlsx", unless user changes this name on line 238.
