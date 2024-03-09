


ARAnet/
│
├── data/                          # Saving a dataset or data file
│   ├── MXene/                     # MXene electrode-related data and results files
│   	├── DOS_data/              # Saving a dataset or data file for DARAnet
│   		├── band_dos/          # DOS data for electrode materials
│   		├── poscar/            # VASP POSCAR for electrode materials
│   ├── MXY-Janus/                 # MXY-Janus electrode-related data and results files
│   ├── TMDC/                      # TMDC electrode-related data and results files
│
├── ui/                    # Standard functions for reading data from the original database
│
├── requirements.txt               # Python libraries that the project depends on
│
├── README.md                      # Main documentation for the project



ARAnet was evaluated in the test set of MXene vertical n-SBH for reproducing the results of the article Figures 4a and 4d. And 5-fold cross-validation results and leave-one-out results are given that differ from the article.The results of all runs can be found in the current directory under the "./data/MXene/" path.
ARAnet_MXene_band_e_test.py

Function modules for input data construction and model construction for DARAnet.
DARAnet_MXene_band_e_test.py