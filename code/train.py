import os
import sys
import torch

import time

import pandas as pd
import numpy as np
import torch


torch.manual_seed(3)
torch.cuda.manual_seed(3)


from active import active_GNN

import warnings
warnings.filterwarnings('ignore')

# from time import clock


def split_train_GNN_onefold(args,logger,path_save):
    seed=34
    for i in range(1):
        # i=args.fold
        path_save_new=path_save+'/fold_'+str(i)+'/'
        try:
            os.mkdir(path_save_new)
        except FileExistsError:
            pass
        args.fold=i
        # for chemprop
        if args.data_path:
            # Get data
            # debug('Loading data')
            args.task_names = args.target_columns or get_task_names(args.data_path)
            data = get_data(path=args.data_path, features_path=args.features_path,args=args)
            # args.num_tasks = data.num_tasks()
            args.features_size = data.features_size()
            # debug(f'Number of tasks = {args.num_tasks}')

            # Split data
            # debug(f'Splitting data with seed {args.seed}')
            if args.separate_test_path:
                test_data = get_data(path=args.separate_test_path, args=args,
                                     features_path=args.separate_test_features_path)
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
            label=None
            if args.mode == 'normal':
                nn=active_GNN(train_data,val_data,test_data,args.mode,path_save_new,['percent_of_unlabel', 1],label,args.fold,False)  # 最后的False 是用来求 first train 用的
                nn.first_train(args,logger)
            else: 
                nn=active_GNN(train_data,val_data,test_data,args.mode,path_save_new,['percent_of_unlabel', 1],label,args.fold,True)  
                nn.train(args,logger)
            # return train_data,val_data,test_data

from args_new import TrainArgs
from chemprop.data.utils import get_class_sizes, get_task_names, split_data,get_data
# from chemprop_utils import get_data
from chemprop.utils import create_logger
vari = {}
fold = False
regression = False
trained = False
# for i in range(int(len(args)/2)):
s_t=time.time()
args = TrainArgs().parse_args()
logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
path_save = args.s
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

split_train_GNN_onefold(args,logger,path_save)
