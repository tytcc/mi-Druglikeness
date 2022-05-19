# Drug-likeness predictor

## Table of Contents

## Requirements

* Python 3.6.13
* rdkit
* chemprop 0.0.2
* alipy 1.2.5
* torch 1.8.0

## Usage 

The whole frame work supports **train**, **prediction**

### Train

```shell
python train.py --data_path ../datasets/marketdrug/worldaintrials_KNIME_rmnu_12-06_train.csv --dataset_type classification --num_workers 2 --save_dir ../pipeline/marketdrug/normal_passive_worldaintrials_KNIME_rmnu_dp2_pretrain_leastc_05-18 --epochs 50 --features_path ../datasets/marketdrug/worldaintrials_KNIME_rmnu_12-06_train_rdkit_2d_normalized.npy --separate_test_path ../datasets/marketdrug/worldaintrials_KNIME_rmnu_12-06_test.csv --separate_test_features_path ../datasets/marketdrug/worldaintrials_KNIME_rmnu_12-06_test_rdkit_2d_normalized.npy --no_features_scaling --metric auc --init_lr 5e-4 --max_lr 5e-4 --final_lr 5e-4 --s ../pipeline/marketdrug/normal_passive_worldaintrials_KNIME_rmnu_dp2_pretrain_leastc_05-18_s --mode passive --depth 2 --target_columns label

```



### Prediction

```shell
python train_2_for_datasets.py --data_path ../datasets/marketdrug/worldaintrials_KNIME_rmnu_12-06_train.csv --dataset_type classification --num_workers 2 --save_dir ../pipeline/marketdrug/worldaintrials_KNIME_rmnu_dp2_leastc_04-13_firsttrain --epochs 50 --features_path ../datasets/marketdrug/worldaintrials_KNIME_rmnu_12-06_train_rdkit_2d_normalized.npy --separate_test_path ../datasets/hepatotoxicity/Hepatotoxicity_KNIME_rmworld.csv --separate_test_features_path ../datasets/hepatotoxicity/Hepatotoxicity_KNIME_rmworld_rdkit_2d_normalized.npy --no_features_scaling --metric auc --init_lr 5e-4 --max_lr 5e-4 --final_lr 5e-4 --s ../pipeline/hepatotoxicity/test_active_worldaintrials_firstrain_hepatotoxicity_KNIME_rmworld_s --mode active --depth 2 --smiles_column smiles --target_columns label 

```

