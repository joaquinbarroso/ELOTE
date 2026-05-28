# ELOTE
Electron Localization of Transition Excitations - Analyzer for Gaussian TD-DFT calculations

What ELOTE does ##
TD-DFT calculations in Gaussian yield a rather obscure output for experimentalists, who prefer a HOMO LUMO labeling to a sequential MO numbering and who wish to have a table with
the percentage contribution of each atomic orbital (AO) to every excited state and thus ascertain whether a transition is internal ligand (IL), Metal-Ligand Charge Transfer (MLCT),
or Ligand-Metal Charge Transfer (LMCT). ELOTE does all this in one run from the composition of atomic orbitals (pop=allorbitals) and the corresponding algebra of the CI coefficients
shown in the TD-DFT output. 

REQUIREMENTS
ELOTE requires python3 and pandas. If run in your terminal you may need to first run:
sudo apt install python3-pandas

USAGE:

python3 ELOTE.py myfile.log

python3 ELOTE.py myfile.log --sort-f           # sort by descending values of f (oscillator strength)

python3 ELOTE.py myfile.log --min-contrib 5.0  # hide transitions < 5%

OUTPUT:
ELOTE displays the Excited States result from Gaussian16, but adds HOMO-n, LUMO+n, labels to each MO
Two files are created:
(1) myfile_ELOTE.csv
An output of each excited state decomposed into their (labeled) MO composition, and a proper assignment if a transiton includes AOs centered on Metal atoms (d and f block elements).
Columns in myfile_ELOTE.csv file are:
lambda calculated (nm), Oscillator Strength (f), Transtion (% contribution), Assignment (if applicable*), E (eV), Excited State (sequential number), Symmetry, S**2 eigenvalue 
(2) myfile_ELOTE_output.txt
An echo of the terminal in text format ready to be copied by the user. Previous to the information present in the csv file, output.txt also contains a modified version of the 
Gaussian output, which includes proper HOMO-i, LUMO+j, labeling and percentage contribution of each transition to the excitation.

Please cite this code and report any issues directly to the developer, Joaquín Barroso from the National Autonomous University of Mexico (UNAM).

Tutorials and blog post:
https://joaquinbarroso.com/ELOTE

