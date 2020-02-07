# Datasets Preprocessing

This repository collects scripts used for various Network IDS datasets preprocessing and analysis.

Preprocessing means:
  - features extraction from PCAPS
  - scaling, train-test split, labeling...

Analsysis means:
  - supervised classification (many learning algorithms)
  - parameters tuning


## Tree

```
.
├── CIC-IDS-2017
│   ├── analysis
│   ├── extra_preprocessing
│   ├── flow_specifications
│   ├── labeling
│   ├── Makefile
│   ├── reproducibility
│   └── statistics
├── LICENSE
└── README.md

```

Each dataset contains the following sub-folders:

  - analysis: contains the supervised analysis scripts
  - extra_preprocessing: contains some extra pre-processing scripts for some feature vectors such as `the multi-key`
  - flow_specifications: contains `json` files used with the [go-flows](https://github.com/CN-TU/go-flows) extractor to extract specific features
  - labeling: scripts for labeling feature vectors based on the dataset documentation
  - reproducibility: reproduce feature vectors extraction experiements
  - statistics: extract some usefull statistics regarding datasets and features such as correlations, frequency tables etc.


## Current Datasets

  - [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)
  - [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)


## Contact
fares.meghdouri@tuwien.ac.at
