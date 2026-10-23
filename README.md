# Decoding-Cell-Death-Electrophysiological-Fingerprints
A reproducible pipeline using BioDecoder to extract electrophysiological fingerprints and classify chemotherapy-induced tumor cell death states from high-throughput microelectrode recordings.


1. Overview
-----------

This package contains supplementary data, feature representations, sample metadata, data-split identifiers, and machine-learning scripts for the study of electrophysiological  lassification of Anlotinib-induced cell-death levels in U87 cells.

The package is organized as follows:

├── README.txt
├── Supplementary_Data/
│   ├── raw_signals/
│   ├── BOW_features/
│   └── preprocessing/
├── Split_Identifiers/
└── Machine_model/
    ├── ElasticNet/
    ├── GBDT/
    ├── RF/
    ├── SVM_Linear/
    └── SVM_rbf/

The directory names shown above are case-sensitive on some operating systems. Please preserve the directory and file names when using the supplied scripts.

2. Supplementary_Data/metadata.csv
----------------------------------

This file contains metadata associated with the released samples and/or recordings. Consult the column headers in the file for the exact fields and their definitions.

Sample identifiers and labels in the metadata should be matched to the corresponding entries in BOW_sample_labels.csv and, where applicable, to the recording or channel identifiers used for the raw signals.

3. Raw electrophysiological signals
-----------------------------------

Location: Supplementary_Data/raw_signals/

Files:
    raw_0_percent.mat
    raw_20_percent.mat
    raw_40_percent.mat
    raw_60_percent.mat

These MATLAB files contain raw electrophysiological recordings associated with the indicated cell-death-level categories. The exact variable names, array dimensions, channel rganization, and recording duration should be checked in each MAT file and against the accompanying metadata.

The percentage labels identify the corresponding experimental categories. They should not be interpreted as a guarantee that every individual recording has an independently measured exact percentage unless this is established by the experimental protocol and metadata.

4. Bag-of-Words (BOW) features
------------------------------

Location: Supplementary_Data/BOW_features/

BOW_feature_matrix.csv
    The numerical feature matrix used to represent samples for machine- learning analysis. Rows should correspond to samples and columns to feature variables; verify the actual orientation and headers in the file.

BOW_feature_names.csv
    The names or identifiers of the BOW feature variables. Use this file to interpret the columns of BOW_feature_matrix.csv.

BOW_sample_labels.csv
    The sample identifiers and class labels associated with the BOW feature matrix. Match rows using the sample identifiers rather than assuming that row order is preserved after external data processing.

For analysis, ensure that each feature-matrix row is matched to the correct sample label. Do not use class labels as predictive input features.

5. Preprocessing feature definition
----------------------------------

Location: Supplementary_Data/preprocessing/

File:
    FixedPeakTypes_top50.mat

This MATLAB file contains the fixed Peak-type feature definitions used in the feature-processing workflow. Inspect the MAT file to confirm the variable names and the exact feature identifiers. Use the same feature definitions when reproducing the corresponding analysis.

6. Split identifiers
--------------------

Location: Split_Identifiers/split_identifiers.csv

This file documents sample assignments for the data-splitting procedure. Consult its column headers and values to identify the dataset, fold, sample identifier, class label, and assignment type recorded in the file.

For cross-validation, distinguish the outer-fold training and test assignments from any inner-fold training and validation assignments used during model selection. A sample may appear in multiple rows because its assignment is recorded separately for different folds or stages.

7. Machine-learning scripts
---------------------------

Location: Machine_model/

The subdirectories contain model-specific training scripts. For example:

    Machine_model/ElasticNet/5-fold-train.py
    Machine_model/GBDT/

Each script implements the corresponding model's training and evaluation workflow. Refer to the individual script for the input file paths, feature loading procedure, cross-validation strategy, hyperparameter-search settings, random seeds, evaluation metrics, and output locations.

Before running a script:
    1. Check that all required data files are present.
    2. Confirm that the paths in the script match the local directory layout.
    3. Confirm that the feature rows and labels are aligned by sample ID.
    4. Confirm that the recorded split assignments match the split procedure
       implemented by the script.
    5. Check the required Python packages and their versions.

Results may vary if software versions, random seeds, input data, or model settings differ from those used in the original analysis.

8. Reproducibility notes
------------------------

- Preserve the released data files and their sample identifiers.
- Do not infer sample IDs, biological replicates, chamber IDs, or chip IDs  from row numbers unless the data documentation explicitly defines that mapping.
- Keep raw recordings, processed features, and labels clearly distinguished.
- Apply feature selection only within the appropriate training data when required by the evaluation design; do not use test-set information to select features or tune model parameters.
- Keep any independent test set separate from cross-validation data.
- If samples share a biological source, ensure that the validation strategy reflects the intended unit of independence and accurately describe the actual splitting method in any report or manuscript.

9. File-specific details
------------------------

The exact MAT-file variable names, array dimensions, metadata fields, units, and independent-test-set assignments are not specified in this README. Please consult the corresponding files and analysis scripts for those details.
