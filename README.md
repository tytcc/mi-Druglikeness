# Subdivisional Druglikeness predictor

This repository contains code used in paper [miDruglikeness](http)

## Requirements

* Python 3.6.13
* rdkit
* chemprop 0.0.2
* alipy 1.2.5
* torch 1.8.0

## Usage 

The whole frame work supports **train**, **prediction**,**al_ensemble**

### Training

To train a model, run:

```shell
python train.py --data_path <train_data> --separate_test_path <test_data> --save_dir <model_path> --s <output> --mode <training mode> 

```

where <train_data> is the path to a CSV file containing training data, <test_data> is the path to a CSV file containing test_data, <model_path> is the directory where trained model will be saved, <output> is the directory where results will be saved, <training mode> is one of "normal", "active" or "passive" for normal training, active learning, or passive learning.

For example:

```python
python train.py --data_path ../datasets/worldaintrials_KNIME_rmnu_06-03_train.csv --dataset_type classification --save_dir ../pipeline/market-approvability_test --epochs 50 --features_path ../datasets/worldaintrials_KNIME_rmnu_06-03_train_rdkit_2d_normalized.npy --separate_test_path ../datasets/worldaintrials_KNIME_rmnu_06-03_test.csv --separate_test_features_path ../datasets/worldaintrials_KNIME_rmnu_06-03_test_rdkit_2d_normalized.npy --no_features_scaling --metric auc --init_lr 5e-4 --max_lr 5e-4 --final_lr 5e-4 --s ../results/test_s --mode normal --depth 2 --hidden_size 300 --ffn_num_layer 2 --target_columns label
```



### Prediction

```shell
python predict.py --separate_test_path <test_data> --save_dir <model_path> --outputfile <outputfile> 

```

where <test_data> is the path to a CSV file containing test data, <model_path> is the directory where trained model is saved, and <outputfile> is the path for prediction results.

For example:

```python
python predict.py --data_path ../datasets/worldaintrials_KNIME_rmnu_06-03_train.csv --dataset_type classification --save_dir ../pipeline/market-approvability --epochs 50 --features_path ../datasets/worldaintrials_KNIME_rmnu_06-03_train_rdkit_2d_normalized.npy --separate_test_path ../datasets/worldaintrials_KNIME_rmnu_06-03_test.csv --separate_test_features_path ../datasets/worldaintrials_KNIME_rmnu_06-03_test_rdkit_2d_normalized.npy --no_features_scaling --metric auc --init_lr 5e-4 --max_lr 5e-4 --final_lr 5e-4 --s ../results/predict_s --mode normal --depth 2 --hidden_size 300 --ffn_num_layer 2 --target_columns label --outputfile output
```



### Al ensemble

```shell
python al_ensemble.py --data_path <train_data> --separate_test_path <test_data> --save_dir <model_path> --s <output> 
```

where <train_data> is the path to a CSV file containing training data, <test_data> is the path to a CSV file containing test_data, <model_path> is the directory where trained model will be ensembled, <output> is the directory where results will be saved.

For example:

```python
python al_ensemble.py --data_path ../datasets/worldaintrials_KNIME_rmnu_06-03_train.csv --dataset_type classification --save_dir ../pipeline/market-approvability --epochs 50 --features_path ../datasets/worldaintrials_KNIME_rmnu_06-03_train_rdkit_2d_normalized.npy --separate_test_path ../datasets/worldaintrials_KNIME_rmnu_06-03_test.csv --separate_test_features_path ../datasets/worldaintrials_KNIME_rmnu_06-03_test_rdkit_2d_normalized.npy --no_features_scaling --metric auc --init_lr 5e-4 --max_lr 5e-4 --final_lr 5e-4 --s ../results/al_ensemble_s --mode normal --depth 2 --hidden_size 300 --ffn_num_layer 2 --target_columns label --start_iter 11 --end_iter 16
```



