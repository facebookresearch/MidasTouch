# TDN: Tactile depth network
Tactile image to heightmap network, based on [FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction) and its PyTorch [implementation](https://github.com/XPFly1989/FCRN). 

## Write a data loading file
- `data/data_to_txt.py` generates dataloader for training/validation/testing given paired image and depth data. To generate this data refer to the `midastouch/data_gen/` folder 

## Train and test model
- `train.py` Trains with the heightmaps and contact masks 
- `test_dataset.py` is testing on cpu. Remember to change the data loading path and result saving path.
