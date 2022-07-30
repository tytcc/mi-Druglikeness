# Drug-likeness predictor

This repository contains code used in paper [Drug-likeness predictor](http)

## Requirements

* Python 3.6.13
* rdkit
* chemprop 0.0.2
* alipy 1.2.5
* torch 1.8.0

## Usage 

The whole frame work supports **train**, **prediction**

### Training

To train a model, run:

```shell
python train.py --data_path <train_data> --separate_test_path <test_data> --save_dir <model_path> --s <output> --mode <training mode> 

```

where <train_data> is the path to a CSV file containing training data, <test_data> is the path to a CSV file containing test_data, <model_path> is the directory where trained model will be saved, <output> is the directory where results will be saved, <training mode> is one of "normal", "active" or "passive" for normal training, active learning, or passive learning.

### Prediction

```shell
python predict.py --separate_test_path <test_data> --save_dir <model_path> --outputfile <outputfile> 

```

where <test_data> is the path to a CSV file containing test data, <model_path> is the directory where trained model is saved, and <outputfile> is the path for prediction results.

### Al ensemble

```shell
python al_ensemble.py --data_path <train_data> --separate_test_path <test_data> --save_dir <model_path> --s <output> 
```

where <train_data> is the path to a CSV file containing training data, <test_data> is the path to a CSV file containing test_data, <model_path> is the directory where trained model will be ensembled, <output> is the directory where results will be saved.
~~we also built a web server for basic predicting. The website is available here:[Druglikeness-Predictor](http)~~

