import os
import sys
import numpy as np
import pandas as pd
import torch

import split_data as split
import preprocessing
import NN
import active
import time
from alipy.query_strategy import (QueryInstanceQBC, QueryInstanceGraphDensity,
                                  QueryInstanceUncertainty, QueryRandom,
                                  QueryCostSensitiveHALC, QueryCostSensitivePerformance,
                                  QueryCostSensitiveRandom)
from alipy import index
from alipy.experiment import StoppingCriteria

from alipy import ToolBox
import copy
import joblib

import pandas as pd
import numpy as np
import torch

# 135
# np.random.seed(33)
torch.manual_seed(3)
torch.cuda.manual_seed(3)

# #160.5
# np.random.seed(33)
# torch.manual_seed(3)
# # torch.cuda.manual_seed(3)

from rdkit.Chem import AllChem
from rdkit import Chem

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors as rdmd
 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from math import sqrt

from chemprop.data import MoleculeDataset

from active_2 import active_GNN,active_GNN_multi
from model_GNN_2 import train_GNN

import warnings
warnings.filterwarnings('ignore')

from time import clock
import pandas as pd

from MOE_my import MoE_model

def get_log(proba,label,prediction, delta = 0.05):
    """ Compute ECE, Average confidence and accuracy for the given
    network on the given dataset """


    bins = [0] * (int(1 // delta) + 1)
    acc_hist = [0] * (int(1 // delta) + 1)
    conf_hist = [0] * (int(1 // delta) + 1)

    correct = 0
    confidence = 0
    for idx in range(0, len(proba)):
        if proba[idx] < 0.5:
            proba_=1-proba[idx]
        else:
            proba_=proba[idx]
        bins[int(proba_ // delta)] += 1
        acc_hist[int(proba_ // delta)] +=1 if prediction[idx] == label[idx] else 0
        conf_hist[int(proba_ // delta)] += proba_
    

    confidence += proba.sum()
    correct +=np.equal(prediction,label).sum() #prediction.eq(label).sum()
    accuracy = 100 * correct/ len(label)
    confidence = 100 * confidence / len(label)
    # print(accuracy)

    s = sum(bins)
    ece = 0
    print(acc_hist)
    print(conf_hist)

    # Normalizations
    for id in range(0, len(bins)):
        acc_hist[id] /= (bins[id] + (bins[id] == 0))
        conf_hist[id] /= (bins[id] + (bins[id] == 0))
        bins[id] /= s
        ece += bins[id] * abs(acc_hist[id] - conf_hist[id])
    return ece, acc_hist,conf_hist


def _label(path):
    suffix = path.split('.')[-1]
    if suffix == 'txt':
        fd = open(path, 'r+')
        lab = fd.read().split('')
        fd.close()
        lab = np.array([float(i) for i in lab])
    else: lab = np.array(joblib.load(path))
    return lab


def _dataset(path):
    suffix = path.split('.')[-1]
    set = None
    if suffix == 'txt' or suffix == 'smi':
        fd = open(path, 'r+')
        set = fd.read().split('')
        fd.close()
        set = preprocessing.smile_list_to_mols(set)
    elif suffix == 'sdf':
        set = preprocessing.read_sdf(path)

    if set:
        set = preprocessing.descriptors.generate_rdDescriptorsSets(set)
    else:
        if suffix == 'csv':
            set = np.array(pd.read_csv(path))
        else: set = np.array(joblib.load(path))

    if suffix == 'npy':
        set = np.load(path)

    return set


def kfold(ds, lab, k=5, regression=False, model=None):
    # torch5Fold
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True)

    fold = 0
    for tr_idx, te_idx in kf.split(lab):
        try:
            os.mkdir('active%d' % fold)
        except FileExistsError:
            pass
        path = 'active%d' % fold
        fold += 1

        xtr, ytr, xte, yte = ds[tr_idx], lab[tr_idx], ds[te_idx], lab[te_idx]
        if model: pass
        else:
            if regression: model = NN.FcRegRDKit()
            else: model = NN.FcRDKit()
        print('start training')
        if regression: nn = active.TorchRegressionFold(xtr, ytr, xte, yte, model, 'active', path, ['percent_of_unlabel', 1],
                                      measure='distance', distance='linear')

        else: nn = active.TorchFold(xtr, ytr, xte, yte, model, 'active', path, ['percent_of_unlabel', 1])
        nn.train()
    print('finish training')


def rm_dup_smi(wdi, zinc):
    '''
    wdi:  negative
    zinc: positive
    '''
    smi_wdi = [Chem.MolToSmiles(x, isomericSmiles=False) for x in wdi.mols()]
    smi_zinc = [Chem.MolToSmiles(x, isomericSmiles=False) for x in zinc.mols()]
    idx = []
    for i in range(len(smi_wdi)):
        if smi_wdi[i] not in smi_zinc:
            idx.append(i)
    temp=np.array(wdi)[idx]
    return MoleculeDataset(temp)

def rm_du(data):
    smi=[Chem.MolToSmiles(x,isomericSmiles=False) for x in data.mols()]
    hash=dict()
    add_idx=[]
    for i,x in enumerate(smi):
        if x not in hash.keys():
            hash[x]=i
            add_idx.append(i)
    temp = np.array(data)[add_idx]
    return MoleculeDataset(temp)

def split_train_GNN_predict(args,logger,path_save):
    seed=34
    # results_all_list=[[],[],[],[],[],[],[],[],[],[]]
    results_all_list=[[],[],[],[],[]]
    for i in range(5):
        args.fold=i
        # i=args.fold
        path_save_new=path_save+'/fold_'+str(i)+'/'
        print(path_save_new)
        try:
            os.mkdir(path_save_new)
        except FileExistsError:
            pass

        # for chemprop
        if args.data_path:
            # Get data
            # debug('Loading data')
            args.task_names = args.target_columns or get_task_names(args.data_path)
            data = get_data(path=args.data_path, features_path=args.features_path,args=args)
            args.num_tasks = data.num_tasks()
            args.features_size = data.features_size()
            debug(f'Number of tasks = {args.num_tasks}')

            # Split data
            # debug(f'Splitting data with seed {args.seed}')
            if args.separate_test_path:
                # test_data = get_data(path=args.separate_test_path, args=args,
                #                      features_path=args.separate_test_features_path,target_columns=['label_world','label_intrials','label_invivo'])
                test_data = get_data(path=args.separate_test_path, args=args,
                                     features_path=args.separate_test_features_path) #target_columns=['label'] smiles_column='smiles'
                args.num_tasks = test_data.num_tasks()
                args.features_size = test_data.features_size()
            if args.separate_val_path:
                val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path)

            if args.separate_val_path and args.separate_test_path:
                train_data = data
                
            elif args.separate_val_path:
                train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.0, 0.2),
                                                      seed=args.seed+i, args=args)
            elif args.separate_test_path:
                train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0),
                                                     seed=seed+i, args=args)
            else:
                train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes,
                                                             seed=seed+i, args=args)

            metric_list=['auc','mcc','gmeans','f1','accuracy','prc-auc']

            # ## debug
            # print(len(test_data))

            # metric_list=['prc-auc']
            len_metric=len(metric_list)
            results = [[] for x in range(len_metric)]
            train_results=[[] for x in range(len_metric)]
            val_results=[[] for x in range(len_metric)]
            test_preds_all = []
            val_preds_all = []
            for round in range(1):
                args.round=round
                net = train_GNN(args)
                # temperature = joblib.load(os.path.join(args.save_dir, f'fold_{args.fold}',f'round_{args.round}','temperature'))
                # temperature=[1]
                # temperature 计算
                temperature=1
                for iteration in range(21,24):#13to16 #15to24 #14to24
                    # test_preds_tem,test_preds,tem=net.predict_GNN_last_tem(test_data,val_data,iteration,temperature,args,logger) #temperature[iteration]
                    # print(len(test_data))
                    # print(np.array(test_preds).shape)
                    test_preds = net.predict_GNN_last(test_data, iteration, args,logger)
                    val_preds = net.predict_GNN_last(val_data, iteration, args,logger)
                    test_preds_all.append(test_preds)
                    val_preds_all.append(val_preds)

                # origin
                test_preds = np.mean(np.array(test_preds_all),axis=0)
                val_preds = np.mean(np.array(val_preds_all),axis=0)


                ## evaluate
                me = net.evaluate_GNN(test_preds, test_data, args, logger, metric_list)
                val_me = net.evaluate_GNN(val_preds, val_data, args, logger, metric_list)



                try:
                    os.mkdir(path_save_new+str(round))
                except FileExistsError:
                    pass
                for mm in range(len_metric):
                    results[mm].append(me[metric_list[mm]])
                    # train_results.append(train_me[metric])
                    val_results[mm].append(val_me[metric_list[mm]])


                    
            for mm in range(len_metric):
                metric = metric_list[mm]
                if metric=='accuracy':
                    metric='acc'
                joblib.dump(results[mm],path_save_new+str(round)+'/'+metric)
                # joblib.dump(train_results, path_save_new + str(round) + '/' +'train_'+ metric)
                # joblib.dump(val_results[mm], path_save_new + str(round) + '/' + 'val_rm_'+metric)
                joblib.dump(val_results[mm], path_save_new + str(round) + '/' + 'val_'+metric)

        results_all_list[i]=test_preds

        # print(np.array(results_all).shape)
        # print(path_save_new+str(round)+'/'+metric)

    return results_all_list



from args import TrainArgs
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.utils import create_logger
vari = {}
fold = False
regression = False
trained = False
# for i in range(int(len(args)/2)):
s_t=time.time()
args = TrainArgs().parse_args()
logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

if args.d:
    path_trainset = args.d
    trainset = _dataset(path_trainset)

if args.l:
    path_trainlab = args.l
    trainlab = _label(path_trainlab)

if args.s:
    path_save = args.s

if args.f:
    k = args.f
    fold = True
elif args.r:
    test_ratio = args.r

if args.R:
    regression = True

if args.m:
    path_model = args[args.index('-m')+1]
    model = torch.load(path_model)
    trained = True

if args.outputfile:
    args.outputfile = './Helloworld/test_example/results/'+args.outputfile


# start training from here
if logger is not None:
    debug, info = logger.debug, logger.info
else:
    debug = info = print
info('Command line')
info(f'python {" ".join(sys.argv)}')

file_path = path_save
try:
    os.mkdir(file_path)
except FileExistsError:
    pass
# os.chdir(file_path)
results=split_train_GNN_predict(args,logger,path_save)

