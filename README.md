# A new approach combining molecular fingerprints and machine learning to estimate relative ionization efficiency in electrospray ionization
### Alfred W. Mayhew, David O. Topping, Jacqueline F. Hamilton ###

The python code presented here was used for the work outlined in "A new approach combining molecular fingerprints and machine learning to estimate relative ionization efficiency in electrospray ionization", Mayhew et al..

The code aims to build a predictive model for the Relative Ionisation Efficiency (RIE) of compounds by encoding compounds as fingerprints (as provided by the UManSysProp package), and testing a range of the machine learning techniques available in the SciKitLearn package.

The data supplied to the models (SMILES structures and measured RIE values) are present in the "RIE-Data" subdirectory. This includes the experimental data collected for the paper, as well as data from "Kruve, A.; Kaupmees, K.; Liigand, J.; Leito, I. Analytical Chemistry 2014, 86, 4822-4830."
