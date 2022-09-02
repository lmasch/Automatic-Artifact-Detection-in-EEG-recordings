# Performance and Reliability Comparison of LSTMs and CNNs for Automatic Classification of Artifacts in EEG Recordings

## Content
#### [Description and Background](#description)
#### [Short Description of the Folder Structure](#navigation)
#### [Virtual Environment](#virtualenv)
#### [Overall Pipeline](#pipeline)

## Description and Background <a name="description"></a>
This repository contains all preprocessing programs and model implementations involved in or needed for my thesis "Performance and Reliability Comparison of LSTMs and CNNs for Automatic Classification of Artifacts in EEG Recordings".

## Short Description of the Folder Structure  <a name="navigation"></a>
In the following I shortly describe where specific files of interest can be found, i.e. which files are contained in which folder.

- **code** \
  The code directory includes the batch files to coordinate the executions of the programs, python scripts for the preprocessing software and jupyter notebooks for the Deep Learning models. Furthermore, the weights, which were obtained during the experiments, are included as well.

- **data** \
  The data directory is an empty directory. Here, once executed, the output of the preprocessing software will be stored in several directories.

- **plots** \
  Here the figures used in the thesis and additional performance plots are provided.

- environment.yml \
  This file is important to install the virtual environment.

- params_04.txt and params_05.txt \
  These text files provide the parameters for the nedc_pystream software and have to stay in the root folder.


## Virtual Environment <a name="virtualenv"></a>
When using Anaconda, open your console and navigate to this folder. In the console type `conda env create -f environment.yml`. To enable this environment, type `conda activate artifact`.


## Overall Pipeline <a name="pipeline"></a>
First, the 'TUH Artifact Corpus' has to be downloaded from the [TUH website](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml). Note, that due to new updates on this corpus, the provided preprocessing programs may no longer work in this configuration. Afterwards, to run the programs and models on your machine, please install the virtual environment using `conda`. To do that, please follow the instructions in the [Section above](#virtualenv).

The data is first of all preprocessed by the 'nedc_pystream' software. To run this program, open your console, activate your environment by typing `conda activate artifact` and change your directory to the root of this repository. Next, execute the nedc_pystream software as indicated below, please provide your own path to the root directory where you stored the downloaded data set as an argument for this execution. Have a look at the usage:
```
usage:
  code\preprocessing_nedc_pystream.bat root

arguments:
  root:               Path to the root folder where the data set has been stored.

Example usage:
  code\preprocessing_nedc_pystream.bat "D:/EEG dataset"
```
This process might take a while. After completing the first preprocessing step, the main preprocessing program needs to be executed. Note, that before running this program, the nedc_pystream software must be executed once. This program requires several arguments according to the intended preprocessing parameters. Have a look at the usage below:
```
usage:
  code\preprocessing_main.bat root export_directory prep std sec overlap exclude num

arguments:
  root:               Path to the root folder where the data set has been stored.
  export_directory:   Path where the output should be exported to.
  prep:               Preprocessing type (either "normal" or "contour").
  std:                Standardization option (choose between "True", "False", "PrevCur", "Entire").
  sec:                Amount of seconds for each segment.
  overlap:            Defines whether the segments should overlap or not (either "True" or "False").
  exclude:            Coefficient to calculate the threshold of exclusion criterion.
  num:                Number of identical values allowed to occur after each other.

Example usage:
  code\preprocessing_main.bat "D:/EEG dataset" "data/prep_normal/example" "normal" "True" 2 "True" 0.02 10

Note:
  Please indicate everything, except for numbers, in parentheses as shown in the example.
```
