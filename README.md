# ASPEN: Spectral-Temporal Fusion for Cross-Subject Brain Decoding
This framework, Adaptive SPectral Encoder Network (ASPEN), aims to improve the cross subject generalization in EEG signals.

## Data Preprocessing
The data used in our analysis and benchmarking was from [MOABB](https://moabb.neurotechx.com/docs/dataset_summary.html).

After downloading the datafiles, to preprocess the data:

      cd data_process
      python preprocess_data.py

## ASPEN
To train (options for task name are 'SSVEP', 'Lee2019_SSVEP', 'BI2014b_P300', 'BNCI2014_P300', 'MI', 'Lee2019_MI'):

      cd model
      python -m train_aspen —task [task name]

## SPEN (SPectral Encoder Network) 
SPEN is only the spectral stream of the network. This doesn't incorporate the temporal signals.

To train (options for task name are 'SSVEP', 'Lee2019_SSVEP', 'BI2014b_P300', 'BNCI2014_P300', 'MI', 'Lee2019_MI'):

      cd model
      python -m train_spen —task [task name]
      
To test (options for task name are 'SSVEP', 'Lee2019_SSVEP', 'BI2014b_P300', 'BNCI2014_P300', 'MI', 'Lee2019_MI'):

      python -m test_spen —task [task name]
      
## Acknowledgements
This work was done as a part of the CMU 11-785: Introduction to Deep Learning course.

## References

### Baselines
- [EEGNet](https://github.com/amrzhd/EEGNet/)  
- [EEGConformer](https://github.com/eeyhsong/EEG-Conformer)
- [TSformer-SA](https://github.com/lixujin99/TSformer-SA) 
- [MultiDiffNet](https://github.com/eddieguo-1128/DualDiff)
- [CTNet](https://github.com/snailpt/CTNet)
### Datasets
- [MOABB](https://moabb.neurotechx.com/docs/dataset_summary.html)