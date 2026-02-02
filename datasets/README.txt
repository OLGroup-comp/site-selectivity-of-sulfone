#Hello and welcome to a brief notice of the files contained in this folder
#Examples of output files can be found in the Output directory

biphenyl_positions.csv
  - Information on Biphenyl systems extracted from CHEMBL database. These excludes heteroatom bicyclic and tricyclic rings on the biphenyl system.
  - Each row contains information on molecular structures: SMILES representation, normalized prinicpal moments of inertia (NMPI) defined across two dimensions, and substitution patterns for
    each ring (excluding the sigma bond that connects the biphenyl system).

dataset.xlsx
 - Excel file used to build our classification model. Currently includes 4 sheets but the predictive model show cased in our paper was built on the "Train ds_add" and "Validation_set_1" sheets
 - Training sheet contains DFT computed ring-opening binary code, descriptors collected along the thiophene core atoms, descriptors based on nucleophiles used.
 - Valdiation sheet contains the same descriptors but binary code based on experimentally observed product.
# Note to user: only a handful of descriptors were used for building the model. Please refer to README file in prinicple directory.

all-figures_nprs-removedNP.csv
 - Information on Biphenyl systems collected from our work. These exclude any biphenyl rings that are associated with natural products.
 - Each row contains information on molecular structures: SMILES representation, and NMPI defined across two dimensions.

chembl_nprs_last.csv
 - Biphenyls (6000 total unique structures) extracted from the CHEMBL database. 
 - NMPI and SMILES are presented for each structure.

cleaned-smiles.csv
 - A small set of biphenyl systems (all unique) that can be used as a practice or as a template.

