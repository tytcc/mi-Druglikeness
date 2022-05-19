#!/Anaconda3/envs/ccjpython/python3
# Created on 2020/12/28 22:17
# Author:tyty
import os
from alipy.query_strategy import QueryInstanceUncertainty

from alipy.query_strategy import QueryInstanceRandom as QueryRandom

from alipy import ToolBox

import joblib



from model_GNN import train_GNN
import pandas as pd
import numpy as np
import torch
np.random.seed(33)
torch.manual_seed(3)
torch.cuda.manual_seed(3)

import time

from sklearn.metrics import confusion_matrix,auc
from sklearn import metrics
from math import sqrt

from chemprop.data import MoleculeDataset
from sklearn.metrics import precision_recall_curve



import warnings
warnings.filterwarnings('ignore')

def auc_prc(targets, preds):
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

 
def update_lr(lr,iteration,n=None,N=None,delta_n=None):

    # return
    return lr
    # return 0.1*(1-n/(1.001*N))
    # if iteration <12:
    #     if iteration % 5 == 0:
    #         lr=lr*0.1
    # else:
    #     if iteration % 5 == 0:
    #         lr=lr*0.5
    # return lr


from copy import deepcopy
def change_None(targets):
    A=deepcopy(targets)
    for i in range(len(targets)):
        for j in range(len(targets[i])):
            if targets[i][j] is None:
                A[i][j]=0
    return A


def update_GNN_lr(iteration,n=None,N=None,delta_n=None):
    initial_lr=5e-4
    lr=initial_lr * ((0.1)**(iteration//5))
    return lr


class active_GNN:
    def __init__(self,train_data,val_data,test_data,phase, path, stopping,cluster_label,fold,activemodel=True):
        np.random.seed(33 + fold)
        self.cluster_label=cluster_label
        self.phase = phase
        # self.classes = int(max(labels))

        if activemodel==True:
            self.alibox = ToolBox(y=change_None(train_data.targets()), query_type='AllLabels', saving_path='./%s' % path)
            self.alibox.split_AL(test_ratio=0, initial_label_rate=0.05, split_count=5)
            if stopping is not None:
                self.stopping_criterion = self.alibox.get_stopping_criterion(stopping[0], value=stopping[1])
            else:
                self.stopping_criterion=self.alibox.get_stopping_criterion()
            self.query_strategy = QueryInstanceUncertainty(X=None, y=None, measure='least_confident')
            self.random = QueryRandom()


        self.unc_result = []
        self.title = ''
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.prc_auc = []
        self.specificity=[]
        self.auc = []
        self.f1 = []
        self.pos = []
        self.neg = []
        self.ratio = []
        self.loss = []
        self.mcc = []
        self.selec=[]
        self.path = path
        self.lr=0.1
        self.num_workers=0

        self.train_data= train_data
        self.val_data=val_data
        self.test_data=test_data

        self.train_acc = []
        self.train_gmeans = []
        self.train_prc_auc = []
        self.train_mcc=[]
        self.train_auc = []
        self.train_f1 = []

        self.val_acc = []
        self.val_gmeans = []
        self.val_prc_auc = []
        self.val_mcc=[]
        self.val_auc = []
        self.val_f1 = []

    def clear(self):
        self.acc = []
        self.gmeans = []
        self.recall = []
        self.precision = []
        self.specificity = []
        self.auc = []
        self.f1 = []
        self.pos = []
        self.neg = []
        self.ratio = []
        self.loss = []
        self.mcc = []
        self.unc_result = []

    def train(self,args,logger):
        test_loss_ceshi = [[], [], [], [], []]
        for round in range(1):
            # round=args.round
            select_idx=[]
            s_t=time.time()
            # self.clear()
            try:
                os.mkdir('%s/%d' % (self.path, round)) # log file
            except FileExistsError:
                pass

            # get data split of one fold
            train_idx, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)

            len_label_ind=len(label_ind)
            len_unlab_ind=len(unlab_ind)

            e_t=time.time()
            print('get split time: {}'.format(e_t-s_t))
            # get intermediate results saver for one fold experiment
            saver = self.alibox.get_stateio(round)

            train_loss_all, test_loss_all ,best_epoch_all,val_loss_all= [],[],[],[]
            # set initial performance point
            # print(torch.cuda.current_device())
            # print(torch.cuda.device_count(), torch.cuda.is_available())
            s_t=time.time()
            args.round=round
            net = train_GNN(args)
            train_data_GNN = MoleculeDataset(np.array(self.train_data)[label_ind])
            iteration = 0

            # ini_train_loss, ini_val_loss, best_epoch = net.pretrain(train_data_GNN, self.val_data,
            #                                                                      iteration, args,
            #                                                                      logger)  # 这里的都是单次GNN 所需要的数据

            # ini_train_loss,ini_val_loss, best_epoch=net.unsupervised_pretrain(train_data_GNN,self.val_data,iteration,args,logger) # 这里的都是单次GNN 所需要的数据
            # args.init_lr = update_GNN_lr(iteration)
            # args.max_lr = update_GNN_lr(iteration)
            # args.final_lr = update_GNN_lr(iteration)
            # ini_train_loss, ini_val_loss, best_epoch,train_results_all,val_results_all = net.train(train_data_GNN, self.val_data, iteration, args,
            #                                                         logger)  # 这里的都是单次GNN 所需要的数据

            ini_train_loss, ini_val_loss, best_epoch,train_results_all,val_results_all = net.train_nopretrain(train_data_GNN, self.val_data, iteration, args,
                                                                    logger)  # 这里的都是单次GNN 所需要的数据

            train_loss_all.append(ini_train_loss)
            val_loss_all.append(ini_val_loss)
            best_epoch_all.append(best_epoch)

            select_idx.append(list(label_ind.index))
            # # evaluate val set
            # val_preds = net.predict_GNN(self.val_data,args,logger)
            # val_results=net.evaluate_GNN(val_preds,self.val_data,args,logger,['mcc','auc','accuracy','f1','gmeans'])

            #evaluate test set
            test_preds = net.predict_GNN(self.test_data,args,logger)
            results = net.evaluate_GNN(test_preds,self.test_data,args,logger,['mcc','prc-auc','accuracy','f1','gmeans','auc','precision','recall','specificity'])

            self.selec.append(len_label_ind)
            self.prc_auc.append(results['prc-auc'])
            self.acc.append(results['accuracy'])
            self.f1.append(results['f1'])
            self.gmeans.append(results['gmeans'])
            self.auc.append(results['auc'])
            self.recall.append(results['recall'])
            self.precision.append(results['precision'])
            self.specificity.append(results['specificity'])
            all = len_label_ind + len_unlab_ind
            lab_init = len_label_ind

            self.mcc.append(results['mcc'])

            self.train_prc_auc.append(train_results_all['prc-auc'])
            self.train_acc.append(train_results_all['accuracy'])
            self.train_f1.append(train_results_all['f1'])
            self.train_gmeans.append(train_results_all['gmeans'])
            self.train_auc.append(train_results_all['auc'])
            self.train_mcc.append(train_results_all['mcc'])

            self.val_prc_auc.append(val_results_all['prc-auc'])
            self.val_acc.append(val_results_all['accuracy'])
            self.val_f1.append(val_results_all['f1'])
            self.val_gmeans.append(val_results_all['gmeans'])
            self.val_auc.append(val_results_all['auc'])
            self.val_mcc.append(val_results_all['mcc'])




            saver.set_initial_point(np.array(results['gmeans']).mean())


            s_all_t=time.time()
            while not self.stopping_criterion.is_stop():

                # select subsets of Uind samples according to query strategy
                iteration += 1

                s_t=time.time()
                if self.phase == 'active':
                    
                    if np.array(unlab_ind).ndim == 2:
                        train_data_GNN_un = MoleculeDataset(np.array(self.train_data)[np.unique(np.array(unlab_ind)[:, 0])])
                    else:
                        train_data_GNN_un = MoleculeDataset(
                            np.array(self.train_data)[unlab_ind])

                    test_preds = net.predict_GNN(train_data_GNN_un, args, logger)

                    for i in range(len(test_preds)):
                        test_preds[i].append(1-test_preds[i][0])
                    prob_pred = tuple(test_preds)
                    if len_label_ind < all*0.5: #0.5 #0.4
                        select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                                predict=prob_pred,
                                                                                # all_unlabel_idx=all_unlabel_ind,
                                                                                batch_size=int(all*0.05)) #all*0.02 # lab_init*0.6 06-07
                                                                                # batch_size=1)

                    else:
                        select_ind = self.query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind,
                                                                                    predict=prob_pred,
                                                                                    # all_unlabel_idx=all_unlabel_ind,
                                                                                    batch_size=int(all*0.1))  ## len_label_ind*0.1 06-07
                                                                                    # batch_size=1)

                elif self.phase == 'passive':
                    if len_label_ind < all*0.5:   # 0.6

                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(all*0.05)) #int(all*0.02) #lab_init*0.6
  
                    else:

                        select_ind = self.random.select(label_ind, unlab_ind, batch_size=int(all*0.1)) #len_label_ind*0.1 06-07

                print('select time {}'.format(time.time()-s_t))


                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                len_label_ind = len(label_ind)
                select_idx.append(list(select_ind))


                s_t=time.time()


                train_data_GNN = MoleculeDataset(np.array(self.train_data)[label_ind])


                # train_loss,val_loss,best_epoch,train_results_all,val_results_all=net.train(train_data_GNN,self.val_data,iteration,args,logger)
                train_loss,val_loss,best_epoch,train_results_all,val_results_all=net.train_nopretrain(train_data_GNN,self.val_data,iteration,args,logger)

                train_loss_all.append(train_loss)
                val_loss_all.append(val_loss)
                best_epoch_all.append(best_epoch)

                print('net train early time: {}'.format(time.time()-s_t))

                test_preds =net.predict_GNN(self.test_data,args,logger)
                results = net.evaluate_GNN(test_preds,self.test_data,args,logger,['mcc','prc-auc','accuracy','f1','gmeans','auc','precision','recall','specificity'])

                self.selec.append(len_label_ind)
                self.prc_auc.append(results['prc-auc'])
                self.auc.append(results['auc'])
                self.acc.append(results['accuracy'])
                self.f1.append(results['f1'])
                self.gmeans.append(results['gmeans'])
                self.recall.append(results['recall'])
                self.precision.append(results['precision'])
                self.specificity.append(results['specificity'])

                self.mcc.append(results['mcc'])

                self.train_prc_auc.append(train_results_all['prc-auc'])
                self.train_acc.append(train_results_all['accuracy'])
                self.train_f1.append(train_results_all['f1'])
                self.train_gmeans.append(train_results_all['gmeans'])
                self.train_auc.append(train_results_all['auc'])
                self.train_mcc.append(train_results_all['mcc'])

                self.val_prc_auc.append(val_results_all['prc-auc'])
                self.val_acc.append(val_results_all['accuracy'])
                self.val_f1.append(val_results_all['f1'])
                self.val_gmeans.append(val_results_all['gmeans'])
                self.val_auc.append(val_results_all['auc'])
                self.val_mcc.append(val_results_all['mcc'])




                # save the results
                st = self.alibox.State(select_ind, np.array(results['gmeans']).mean())
                saver.add_state(st)
                saver.save()



                self.stopping_criterion.update_information(saver)



            print('s_all_t time: {}'.format(time.time()-s_all_t))

            self.stopping_criterion.reset()

            joblib.dump(net.temperature,os.path.join(net.save_dir, 'temperature'))
            joblib.dump(select_idx,'./%s/%d/select_idx' % (self.path, round))
            joblib.dump(self.prc_auc, './%s/%d/prc-auc' % (self.path, round))
            joblib.dump(self.auc, './%s/%d/auc' % (self.path, round))
            joblib.dump(self.acc, './%s/%d/acc' % (self.path, round))
            joblib.dump(self.f1, './%s/%d/f1' % (self.path, round))
            joblib.dump(self.gmeans, './%s/%d/gmeans' % (self.path, round))
            joblib.dump(self.recall, './%s/%d/recall' % (self.path, round))
            joblib.dump(self.precision, './%s/%d/precision' % (self.path, round))
            joblib.dump(self.specificity, './%s/%d/specificity' % (self.path, round))
            joblib.dump(self.pos, './%s/%d/pos' % (self.path, round))
            joblib.dump(self.neg, './%s/%d/neg' % (self.path, round))
            joblib.dump(self.ratio, './%s/%d/ratio' % (self.path, round))
            joblib.dump(self.mcc, './%s/%d/mcc' % (self.path, round))
            joblib.dump(train_loss_all,'./%s/%d/train_loss' %  (self.path, round))
            joblib.dump(val_loss_all, './%s/%d/val_loss' % (self.path, round))
            joblib.dump(best_epoch_all,'./%s/%d/best_epoch' % (self.path, round))
            joblib.dump(self.selec, './%s/%d/select' % (self.path, round))
            joblib.dump(all,'./%s/%d/all'  % (self.path, round))

            joblib.dump(self.train_prc_auc, './%s/%d/train_prc-auc' % (self.path, round))
            joblib.dump(self.train_auc, './%s/%d/train_auc' % (self.path, round))
            joblib.dump(self.train_acc, './%s/%d/train_acc' % (self.path, round))
            joblib.dump(self.train_f1, './%s/%d/train_f1' % (self.path, round))
            joblib.dump(self.train_gmeans, './%s/%d/train_gmeans' % (self.path, round))
            joblib.dump(self.train_mcc, './%s/%d/train_mcc' % (self.path, round))

            joblib.dump(self.val_prc_auc, './%s/%d/val_prc-auc' % (self.path, round))
            joblib.dump(self.val_auc, './%s/%d/val_auc' % (self.path, round))
            joblib.dump(self.val_acc, './%s/%d/val_acc' % (self.path, round))
            joblib.dump(self.val_f1, './%s/%d/val_f1' % (self.path, round))
            joblib.dump(self.val_gmeans, './%s/%d/val_gmeans' % (self.path, round))
            joblib.dump(self.val_mcc, './%s/%d/val_mcc' % (self.path, round))





        self.test_loss_ceshi=test_loss_ceshi

        return train_loss_all,val_loss_all,self

    ## training without active learning
    def first_train(self,args,logger):
        test_loss_ceshi = [[], [], [], [], []]
        for round in range(1):
            args.round = round
            # round=args.round
            s_t=time.time()
            # self.clear()
            try:
                os.mkdir('%s/%d' % (self.path, round)) # log file
            except FileExistsError:
                pass


            train_loss_all, test_loss_all ,best_epoch_all,val_loss_all= [],[],[],[]

            s_t=time.time()
            net = train_GNN(args)
            train_data_GNN = MoleculeDataset(np.array(self.train_data))
            iteration=0
            ini_train_loss,ini_val_loss, best_epoch,train_results_all,val_results_all=net.train(train_data_GNN,self.val_data,iteration,args,logger) # 这里的都是单次GNN 所需要的数据

            train_loss_all.append(ini_train_loss)
            val_loss_all.append(ini_val_loss)
            best_epoch_all.append(best_epoch)

            # # evaluate val set
            # val_preds = net.predict_GNN(self.val_data,args,logger)
            # val_results=net.evaluate_GNN(val_preds,self.val_data,args,logger,['mcc','auc','accuracy','f1','gmeans'])

            #evaluate test set
            test_preds = net.predict_GNN(self.test_data,args,logger)
            results = net.evaluate_GNN(test_preds,self.test_data,args,logger,['mcc','prc-auc','accuracy','f1','gmeans','auc'])  #'precision','recall','specificity'



            self.prc_auc.append(results['prc-auc'])
            self.auc.append(results['auc'])
            self.acc.append(results['accuracy'])
            self.f1.append(results['f1'])
            self.gmeans.append(results['gmeans'])

            self.mcc.append(results['mcc'])

            self.train_prc_auc.append(train_results_all['prc-auc'])
            self.train_acc.append(train_results_all['accuracy'])
            self.train_f1.append(train_results_all['f1'])
            self.train_gmeans.append(train_results_all['gmeans'])
            self.train_auc.append(train_results_all['auc'])
            self.train_mcc.append(train_results_all['mcc'])

            self.val_prc_auc.append(val_results_all['prc-auc'])
            self.val_acc.append(val_results_all['accuracy'])
            self.val_f1.append(val_results_all['f1'])
            self.val_gmeans.append(val_results_all['gmeans'])
            self.val_auc.append(val_results_all['auc'])
            self.val_mcc.append(val_results_all['mcc'])

            iteration = 0
            s_all_t=time.time()

            joblib.dump(net.temperature,os.path.join(net.save_dir, 'temperature'))
            joblib.dump(self.prc_auc, './%s/%d/prc_auc' % (self.path, round))
            joblib.dump(self.auc, './%s/%d/auc' % (self.path, round))
            joblib.dump(self.acc, './%s/%d/acc' % (self.path, round))
            joblib.dump(self.f1, './%s/%d/f1' % (self.path, round))
            joblib.dump(self.gmeans, './%s/%d/gmeans' % (self.path, round))
            joblib.dump(self.recall, './%s/%d/recall' % (self.path, round))
            joblib.dump(self.precision, './%s/%d/precision' % (self.path, round))
            joblib.dump(self.specificity, './%s/%d/specificity' % (self.path, round))
            joblib.dump(self.pos, './%s/%d/pos' % (self.path, round))
            joblib.dump(self.neg, './%s/%d/neg' % (self.path, round))
            joblib.dump(self.ratio, './%s/%d/ratio' % (self.path, round))
            joblib.dump(self.mcc, './%s/%d/mcc' % (self.path, round))
            joblib.dump(train_loss_all,'./%s/%d/train_loss' %  (self.path, round))
            joblib.dump(val_loss_all, './%s/%d/val_loss' % (self.path, round))
            joblib.dump(best_epoch_all,'./%s/%d/best_epoch' % (self.path, round))
            joblib.dump(self.selec, './%s/%d/select' % (self.path, round))
            joblib.dump(all,'./%s/%d/all'  % (self.path, round))

            joblib.dump(self.val_prc_auc, './%s/%d/val_prc-auc' % (self.path, round))
            joblib.dump(self.val_auc, './%s/%d/val_auc' % (self.path, round))
            joblib.dump(self.val_acc, './%s/%d/val_acc' % (self.path, round))
            joblib.dump(self.val_f1, './%s/%d/val_f1' % (self.path, round))
            joblib.dump(self.val_gmeans, './%s/%d/val_gmeans' % (self.path, round))
            joblib.dump(self.val_mcc, './%s/%d/val_mcc' % (self.path, round))

            joblib.dump(self.train_prc_auc, './%s/%d/train_prc-auc' % (self.path, round))
            joblib.dump(self.train_auc, './%s/%d/train_auc' % (self.path, round))
            joblib.dump(self.train_acc, './%s/%d/train_acc' % (self.path, round))
            joblib.dump(self.train_f1, './%s/%d/train_f1' % (self.path, round))
            joblib.dump(self.train_gmeans, './%s/%d/train_gmeans' % (self.path, round))
            joblib.dump(self.train_mcc, './%s/%d/train_mcc' % (self.path, round))

        return train_loss_all,test_loss_all,self
