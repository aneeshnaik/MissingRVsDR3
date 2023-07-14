# MissingRVsDR3

## Summary

This repository contains the code and the trained model used to generate the results in the article Naik & Widmark (2023; arXiv link to be added soon): "The missing radial velocities of \textit{Gaia}: a catalogue of Bayesian estimates for DR3."

The code includes the data queries used to obtain the raw Gaia data, the scripts used to train the Bayesian neural network, the scripts to generate the catalogue, and the scripts used to generate the plots in the paper. 

The Bayesian neural network source code is available as the package [banyan](https://github.com/aneeshnaik/banyan)

## Citation

Our code is freely available for use under the MIT License. For details, see LICENSE.

If using our code, please cite our paper: arXiv link to be added soon.


## Structure

This code is structured as follows:
- `/src` contains all of the 'source code', including utility functions (`utils.py`), coordinate conversions (`coords.py`), global constants (`constants.py`), run parameters (`params.py`), and various functions relating to the Gaussian mixture models (`gmm.py`).
- `/EDR3_match` contains a script used to match stars contained in our previous EDR3-derived catalogue to stars with measured radial velocities in DR3.
- `/data` contains a directory of data queries, as well as various scripts used to parse and manage the downloaded data.
- `/models` contains the script used to train the BNN model, as well as the ensemble of trained BNNs itself (`*.pth`).
- `/predictions` contains the scripts used to generate predictions from the trained BNN model, as well as scripts to fit the prediction distributions with Gaussian mixture models.
- `/figures` contains the plotting scripts used for the paper.
