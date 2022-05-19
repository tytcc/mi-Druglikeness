#!/Anaconda3/envs/ccjpython/python3
# Created on 2021/2/1 22:09
# Author:tyty
# Graph model

from rdkit import Chem
from chemprop.nn_utils import index_select_ND, get_activation_function
import numpy as np
from chemprop.features.featurization import BatchMolGraph,mol2graph
from chemprop.data import StandardScaler, MoleculeDataLoader

import torch
import torch.nn as nn

import os
from typing import List,Union
from logging import Logger
import logging



import numpy as np
import torch

torch.manual_seed(3)
torch.cuda.manual_seed(3)

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR

# from chemprop.train.evaluate import evaluate
from chemprop_evaluate import evaluate_predictions, multievaluate,evaluate_predictions_list,multipredict,evaluate,temperature_predict
from chemprop.train.predict import predict
from chemprop_train import train,multitrain,temperature_train
from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, MoleculeDataLoader
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import MoleculeModel
from chemprop_model import MultiConcatModel,resMoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits

from torch.optim import Adam, Optimizer
def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """

    params = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.init_lr, 'weight_decay': 0}]

    return Adam(params)

### utils
def load_checkpoint(path: str,
                    device: torch.device = None,
                    m_type: str = 'single',
                    logger: logging.Logger = None) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state['args']), skip_unsettable=True)
    loaded_state_dict = state['state_dict']

    if device is not None:
        args.device = device

    # Build model
    if m_type == 'multi':
        model = MultiConcatModel(args,output=[args.sou_num_tasks,args.num_tasks])
        model_state_dict = model.state_dict()
    elif m_type == 'single':
        model = DMPNN(args)
        model_state_dict = model.state_dict()
    elif m_type == 'res':
        model = resMoleculeModel(args)
        model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            info(f'Warning: Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            info(f'Warning: Pretrained parameter "{param_name}" '
                 f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug('Moving model to cuda')
    model = model.to(args.device)

    return model




MAX_ATOMIC_NUM = 100
ATOM_FEATURES = \
    {
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-1, -2, 1, 2, 0],
        'chiral_tag': [0, 1, 2, 3],
        'num_Hs': [0, 1, 2, 3, 4],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ],
    }

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2  # 133
BOND_FDIM = 14


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,args=None):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param atom_messages: Whether to use atoms to pass messages instead of bonds.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM + ATOM_FDIM
        self.atom_messages = args.atom_messages  # 是否只使用atom的信息
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        # self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function('ReLU')

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if type(mol_graph) != BatchMolGraph:
            mol_graph = mol2graph(mol_graph)

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(
            atom_messages=self.atom_messages)
        # f_atoms, f_bonds, a2b, b2a, b2revb = mol_graph.get_components(
        #     atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(
            self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size  #此处的bonds的特征已经将atom和bond的特征拼在了一起
        message = self.act_func(input)  # num_bonds x hidden_size   num_bonds x (133+14) -> 300 hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2  # 就是把两个对应的平均一下从而可以得到一个无偏的值

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds),
                                        dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message,
                                                a2b)  # num_atoms x max_num_bonds x hidden  #把bondsmessage 转成了 atom的
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden #为啥要减去 相反的键的数据聚合的信息

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        ###这是最后一层的聚合
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)  # 按分子数 分配好
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size+features)

        # return atom_hiddens
        return mol_vecs

def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

class fn(torch.nn.Module):
    def __init__(self,args,output=None):
        super(fn,self).__init__()
        self.ffn_num_layers=args.ffn_num_layers #args.ffn_num_layers
        self.ffn_hidden_size=args.ffn_hidden_size #args.ffn_hidden_size
        if output is not None:
            self.output_size = output
        else:
            self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.create_ffn(args)
        initialize_weights(self)

    def create_ffn(self, args: TrainArgs):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.ffn_hidden_size #args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function('ReLU') #args.activation

        # Create FFN layers
        if self.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(self.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input, train_phase=True):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """

        output = self.ffn(*input)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if train_phase:
            if self.classification and not self.training:
                output = self.sigmoid(output)

        if self.multiclass:
            output = output.reshape(
                (output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(
                    output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


class DMPNN(torch.nn.Module):

    def __init__(self, args: TrainArgs, featurizer: bool = False, output=None):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e. outputting
                           learned features in the final layer before prediction.
        """
        super(DMPNN, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        if output is not None:
            self.output_size = output
        else:
            self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)


        self.ffn_num_layers=args.ffn_num_layers #args.ffn_num_layers
        self.ffn_hidden_size=args.ffn_hidden_size #args.ffn_hidden_size
        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)


    def create_encoder(self, args: TrainArgs):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPNEncoder(args=args)

    def create_ffn(self, args: TrainArgs):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.ffn_hidden_size #args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function('ReLU') #args.activation

        # Create FFN layers
        if self.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(self.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def featurize(self, *input):
        """
        Computes feature vectors of the input by leaving out the last layer.
        :param input: Input.
        :return: The feature vectors computed by the MoleculeModel.
        """
        return self.ffn[:-1](self.encoder(*input))

    def forward(self, *input,train_phase=True):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """
        if self.featurizer:
            return self.featurize(*input)  # 只计算graph model 计算出的encoder的编码结果

        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if train_phase:
            if self.classification and not self.training:
                output = self.sigmoid(output)

        if self.multiclass:
            output = output.reshape(
                (output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(
                    output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


class train_GNN():
    def __init__(self,args):
        self.num_workers = 0
        # Get loss and metric functions
        self.loss_func = get_loss_func(args)
        self.metric_func = get_metric_func(metric=args.metric)
        self.save_dir_root = os.path.join(args.save_dir, f'fold_{args.fold}')
        self.temperature=[]

    def train(self,train_data,val_data,iteration,args,logger):
        self.loss_func = get_loss_func(args) # normal loss function
        # self.loss_func = FocalLoss_GNN(gamma=0.7,alpha=None)
        cache = False
        args.train_data_size = len(train_data)
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print
        info(f'iteration {iteration}')
        if args.dataset_type == 'classification':
            class_sizes = get_class_sizes(train_data)
            debug('Class sizes')
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

        # if args.save_smiles_splits:
        #     save_smiles_splits(
        #         train_data=train_data,
        #         val_data=val_data,
        #         test_data=test_data,
        #         data_path=args.data_path,
        #         save_dir=args.save_dir
        #     )

        if args.features_scaling:
            # 对feature 进行normalize
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            # test_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        self.features_scaler=features_scaler
        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        if args.dataset_type == 'regression':
            debug('Fitting scaler')
            train_smiles, train_targets = train_data.smiles(), train_data.targets()
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
            train_data.set_targets(scaled_targets)
        else:
            scaler = None
        self.scaler=scaler
        # # Set up test set evaluation
        # test_smiles, test_targets = test_data.smiles(), test_data.targets()
        # if args.dataset_type == 'multiclass':
        #     sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        # else:
        #     sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))


        # Create data loaders
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache,
            # class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )

        train_show_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache,
        )

        val_data_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache
        )
        # test_data_loader = MoleculeDataLoader(
        #     dataset=test_data,
        #     batch_size=args.batch_size,
        #     num_workers=self.num_workers,
        #     cache=cache
        # )

        # # Tensorboard writer
        model_idx=0
        if args.round==None:
            save_dir = self.save_dir_root

        else:
            save_dir = os.path.join(self.save_dir_root, f'round_{args.round}')


        self.save_dir=save_dir

        makedirs(save_dir)

        # try:
        #     writer = SummaryWriter(log_dir=save_dir)
        # except:
        #     writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if os.path.exists(os.path.join(save_dir, 'model.pt')):
            # debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            # model = load_checkpoint(os.path.join(save_dir, 'model.pt'), logger=logger)
            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = DMPNN(args)
        # record the iteration model on the begining
        save_checkpoint(os.path.join(save_dir, 'model_iteration_'+str(iteration)+'.pt'), model, scaler, features_scaler, args)


        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        # save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0

        list_n_iter = []
        train_loss = []
        param_norm = []
        gradient_norm = []
        val_loss = []
        for epoch in trange(args.epochs):
            info(f'Epoch {epoch}')
            n_iter, l_n_iter, a, b, c = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=self.loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                # writer=writer
            )

            list_n_iter.append(l_n_iter)
            train_loss.append(a)
            param_norm.append(b)
            gradient_norm.append(c)

            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores, val_R2,val_score_all = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metric_func=self.metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            # evaluate train_set
            train_scores, train_R2,train_scores_all = evaluate(
                model=model,
                data_loader=train_show_data_loader,
                num_tasks=args.num_tasks,
                metric_func=self.metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=None
            )
            #
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            # writer.add_scalar(f'validation_{args.metric}', avg_val_score, epoch)
            # # 加入 val_loss
            # writer.add_scalar(f'validation_loss', np.nanmean(val_R2[1]), epoch)
            val_loss.append(np.nanmean(val_R2[1]))
            # # 加入 train_auc
            # writer.add_scalar(f'train_{args.metric}', np.nanmean(train_scores), epoch)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    # writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, epoch)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
                # 计算 best epoch里的train_auc
                best_train_score = np.nanmean(train_scores)

            # Evaluate on test set using model with best validation score
#             info(f'Model {model_idx} best train {args.metric} = {best_train_score:.6f} on epoch {best_epoch}')
#             info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            ## 现在没有ensemble 所以 先用round 来代替
            info(f'round {args.round} best train {args.metric} = {best_train_score:.6f} on epoch {best_epoch}')
            info(f'round {args.round} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            # model = load_checkpoint(os.path.join(save_dir, 'model.pt'), device=args.device, logger=logger)

            # test_preds = predict(
            #     model=model,
            #     data_loader=test_data_loader,
            #     scaler=scaler
            # )
            # test_scores, test_R2 = evaluate_predictions(
            #     preds=test_preds,
            #     targets=test_targets,
            #     num_tasks=args.num_tasks,
            #     metric_func=metric_func,
            #     dataset_type=args.dataset_type,
            #     logger=logger
            # )

        #     if len(test_preds) != 0:
        #         sum_test_preds += np.array(test_preds)
        #
        #     # Average test score
        #     avg_test_score = np.nanmean(test_scores)
        #     info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        #     writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)
        #
        #     if args.show_individual_scores:
        #         # Individual test scores
        #         for task_name, test_score in zip(args.task_names, test_scores):
        #             info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
        #             writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)
        #
        # # Evaluate ensemble on test set
        # avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
        #
        # ensemble_scores, ensemble_R2 = evaluate_predictions(
        #     preds=avg_test_preds,
        #     targets=test_targets,
        #     num_tasks=args.num_tasks,
        #     metric_func=metric_func,
        #     dataset_type=args.dataset_type,
        #     logger=logger
        # )
        #
        # # Average ensemble score
        # avg_ensemble_test_score = np.nanmean(ensemble_scores)
        # info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
        # writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)
        #
        # # Individual ensemble scores
        # if args.show_individual_scores:
        #     for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
        #         info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
        #temperature 计算
        if os.path.exists(os.path.join(save_dir, 'model.pt')):
            # debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), logger=logger)
            tem= temperature_train(
            model=model,
            data_loader=val_data_loader,
            loss_func=self.loss_func,
            args=args,)
            self.temperature.append(tem)

        return train_loss,val_loss,best_epoch,train_scores_all,val_score_all

    def train_nopretrain(self, train_data, val_data, iteration, args, logger):
        self.loss_func = get_loss_func(args)  # normal loss function
        # self.loss_func = FocalLoss_GNN(gamma=0.7,alpha=None)
        cache = False
        args.train_data_size = len(train_data)
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print
        info(f'iteration {iteration}')
        if args.dataset_type == 'classification':
            class_sizes = get_class_sizes(train_data)
            debug('Class sizes')
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

        # if args.save_smiles_splits:
        #     save_smiles_splits(
        #         train_data=train_data,
        #         val_data=val_data,
        #         test_data=test_data,
        #         data_path=args.data_path,
        #         save_dir=args.save_dir
        #     )

        if args.features_scaling:
            # 对feature 进行normalize
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            # test_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        self.features_scaler = features_scaler
        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        if args.dataset_type == 'regression':
            debug('Fitting scaler')
            train_smiles, train_targets = train_data.smiles(), train_data.targets()
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
            train_data.set_targets(scaled_targets)
        else:
            scaler = None
        self.scaler = scaler
        # # Set up test set evaluation
        # test_smiles, test_targets = test_data.smiles(), test_data.targets()
        # if args.dataset_type == 'multiclass':
        #     sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        # else:
        #     sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        # Create data loaders
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache,
            # class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )

        train_show_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache,
        )

        val_data_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache
        )
        # test_data_loader = MoleculeDataLoader(
        #     dataset=test_data,
        #     batch_size=args.batch_size,
        #     num_workers=self.num_workers,
        #     cache=cache
        # )

        # # Tensorboard writer
        model_idx = 0
        if args.round == None:
            save_dir = self.save_dir_root

        else:
            save_dir = os.path.join(self.save_dir_root, f'round_{args.round}')

        self.save_dir = save_dir

        makedirs(save_dir)

        # try:
        #     writer = SummaryWriter(log_dir=save_dir)
        # except:
        #     writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if os.path.exists(os.path.join(save_dir, 'model.pt')):
            # debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            # model = load_checkpoint(os.path.join(save_dir, 'model.pt'), logger=logger)
            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), logger=logger)
            save_checkpoint(os.path.join(save_dir, 'model_iteration_' + str(iteration) + '.pt'), model, scaler,
                            features_scaler, args)
            initialize_weights(model)
        else:
            debug(f'Building model {model_idx}')
            model = DMPNN(args)
            save_checkpoint(os.path.join(save_dir, 'model_iteration_' + str(iteration) + '.pt'), model, scaler,
                            features_scaler, args)
        # record the iteration model on the begining


        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        # save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0

        list_n_iter = []
        train_loss = []
        param_norm = []
        gradient_norm = []
        val_loss = []
        for epoch in trange(args.epochs):
            info(f'Epoch {epoch}')
            n_iter, l_n_iter, a, b, c = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=self.loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                # writer=writer
            )

            list_n_iter.append(l_n_iter)
            train_loss.append(a)
            param_norm.append(b)
            gradient_norm.append(c)

            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores, val_R2, val_score_all = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metric_func=self.metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            # evaluate train_set
            train_scores, train_R2, train_scores_all = evaluate(
                model=model,
                data_loader=train_show_data_loader,
                num_tasks=args.num_tasks,
                metric_func=self.metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=None
            )
            #
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            # writer.add_scalar(f'validation_{args.metric}', avg_val_score, epoch)
            # # 加入 val_loss
            # writer.add_scalar(f'validation_loss', np.nanmean(val_R2[1]), epoch)
            val_loss.append(np.nanmean(val_R2[1]))
            # # 加入 train_auc
            # writer.add_scalar(f'train_{args.metric}', np.nanmean(train_scores), epoch)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    # writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, epoch)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
                # 计算 best epoch里的train_auc
                best_train_score = np.nanmean(train_scores)

            # Evaluate on test set using model with best validation score
            #             info(f'Model {model_idx} best train {args.metric} = {best_train_score:.6f} on epoch {best_epoch}')
            #             info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            ## 现在没有ensemble 所以 先用round 来代替
            info(f'round {args.round} best train {args.metric} = {best_train_score:.6f} on epoch {best_epoch}')
            info(f'round {args.round} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            # model = load_checkpoint(os.path.join(save_dir, 'model.pt'), device=args.device, logger=logger)

            # test_preds = predict(
            #     model=model,
            #     data_loader=test_data_loader,
            #     scaler=scaler
            # )
            # test_scores, test_R2 = evaluate_predictions(
            #     preds=test_preds,
            #     targets=test_targets,
            #     num_tasks=args.num_tasks,
            #     metric_func=metric_func,
            #     dataset_type=args.dataset_type,
            #     logger=logger
            # )

        #     if len(test_preds) != 0:
        #         sum_test_preds += np.array(test_preds)
        #
        #     # Average test score
        #     avg_test_score = np.nanmean(test_scores)
        #     info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        #     writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)
        #
        #     if args.show_individual_scores:
        #         # Individual test scores
        #         for task_name, test_score in zip(args.task_names, test_scores):
        #             info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
        #             writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)
        #
        # # Evaluate ensemble on test set
        # avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
        #
        # ensemble_scores, ensemble_R2 = evaluate_predictions(
        #     preds=avg_test_preds,
        #     targets=test_targets,
        #     num_tasks=args.num_tasks,
        #     metric_func=metric_func,
        #     dataset_type=args.dataset_type,
        #     logger=logger
        # )
        #
        # # Average ensemble score
        # avg_ensemble_test_score = np.nanmean(ensemble_scores)
        # info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
        # writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)
        #
        # # Individual ensemble scores
        # if args.show_individual_scores:
        #     for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
        #         info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
        # temperature 计算
        if os.path.exists(os.path.join(save_dir, 'model.pt')):
            # debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), logger=logger)
            tem = temperature_train(
                model=model,
                data_loader=val_data_loader,
                loss_func=self.loss_func,
                args=args, )
            self.temperature.append(tem)

        return train_loss, val_loss, best_epoch, train_scores_all, val_score_all

    def succ_train(self, train_data, val_data, iteration, args, logger):
        self.loss_func = get_loss_func(args)  # normal loss function
        # self.loss_func = FocalLoss_GNN(gamma=0.7,alpha=None)
        cache = False
        args.train_data_size = len(train_data)
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print
        info(f'iteration {iteration}')
        if args.dataset_type == 'classification':
            class_sizes = get_class_sizes(train_data)
            debug('Class sizes')
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

        # if args.save_smiles_splits:
        #     save_smiles_splits(
        #         train_data=train_data,
        #         val_data=val_data,
        #         test_data=test_data,
        #         data_path=args.data_path,
        #         save_dir=args.save_dir
        #     )

        if args.features_scaling:
            # 对feature 进行normalize
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            # test_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        self.features_scaler = features_scaler
        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        if args.dataset_type == 'regression':
            debug('Fitting scaler')
            train_smiles, train_targets = train_data.smiles(), train_data.targets()
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
            train_data.set_targets(scaled_targets)
        else:
            scaler = None
        self.scaler = scaler
        # # Set up test set evaluation
        # test_smiles, test_targets = test_data.smiles(), test_data.targets()
        # if args.dataset_type == 'multiclass':
        #     sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        # else:
        #     sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        # Create data loaders
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache,
            # class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )

        train_show_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache,
        )

        val_data_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache
        )
        # test_data_loader = MoleculeDataLoader(
        #     dataset=test_data,
        #     batch_size=args.batch_size,
        #     num_workers=self.num_workers,
        #     cache=cache
        # )

        # # Tensorboard writer
        model_idx = 0
        save_dir = os.path.join(self.save_dir_root, f'round_{args.round}')
        self.save_dir = save_dir
        makedirs(save_dir)
        # try:
        #     writer = SummaryWriter(log_dir=save_dir)
        # except:
        #     writer = SummaryWriter(logdir=save_dir)

        if os.path.exists(os.path.join(save_dir,'model_iteration_' + str(iteration+1) + '.pt')):
            model=load_checkpoint(os.path.join(save_dir, 'model_iteration_'+str(iteration+1)+'.pt'), logger=logger)
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
            return [],[],[]
        else:


            # Load/build model
            if os.path.exists(os.path.join(save_dir, 'model.pt')):
                # debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
                model = load_checkpoint(os.path.join(save_dir, 'model.pt'), logger=logger)
            else:
                debug(f'Building model {model_idx}')
                model = DMPNN(args)
            # record the iteration model
            save_checkpoint(os.path.join(save_dir, 'model_iteration_' + str(iteration) + '.pt'), model, scaler,
                            features_scaler, args)



            debug(model)
            debug(f'Number of parameters = {param_count(model):,}')
            if args.cuda:
                debug('Moving model to cuda')
            model = model.to(args.device)

            # Ensure that model is saved in correct location for evaluation if 0 epochs
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            # Optimizers
            optimizer = build_optimizer(model, args)

            # Learning rate schedulers
            scheduler = build_lr_scheduler(optimizer, args)

            # Run training
            best_score = float('inf') if args.minimize_score else -float('inf')
            best_epoch, n_iter = 0, 0

            list_n_iter = []
            train_loss = []
            param_norm = []
            gradient_norm = []
            val_loss = []
            for epoch in trange(args.epochs):
                info(f'Epoch {epoch}')
                n_iter, l_n_iter, a, b, c = train(
                    model=model,
                    data_loader=train_data_loader,
                    loss_func=self.loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    logger=logger,
                    # writer=writer
                )

                list_n_iter.append(l_n_iter)
                train_loss.append(a)
                param_norm.append(b)
                gradient_norm.append(c)

                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
                val_scores, val_R2, _ = evaluate(
                    model=model,
                    data_loader=val_data_loader,
                    num_tasks=args.num_tasks,
                    metric_func=self.metric_func,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    logger=logger
                )
                # evaluate train_set
                train_scores, train_R2, _ = evaluate(
                    model=model,
                    data_loader=train_show_data_loader,
                    num_tasks=args.num_tasks,
                    metric_func=self.metric_func,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    logger=None
                )
                #
                # Average validation score
                avg_val_score = np.nanmean(val_scores)
                debug(f'Validation {args.metric} = {avg_val_score:.6f}')
                # writer.add_scalar(f'validation_{args.metric}', avg_val_score, epoch)
                # # 加入 val_loss
                # writer.add_scalar(f'validation_loss', np.nanmean(val_R2[1]), epoch)
                val_loss.append(np.nanmean(val_R2[1]))
                # # 加入 train_auc
                # writer.add_scalar(f'train_{args.metric}', np.nanmean(train_scores), epoch)

                if args.show_individual_scores:
                    # Individual validation scores
                    for task_name, val_score in zip(args.task_names, val_scores):
                        debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                        # writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, epoch)

                # Save model checkpoint if improved validation score
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
                    # 计算 best epoch里的train_auc
                    best_train_score = np.nanmean(train_scores)

                # Evaluate on test set using model with best validation score
                #             info(f'Model {model_idx} best train {args.metric} = {best_train_score:.6f} on epoch {best_epoch}')
                #             info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
                ## 现在没有ensemble 所以 先用round 来代替
                info(f'round {args.round} best train {args.metric} = {best_train_score:.6f} on epoch {best_epoch}')
                info(f'round {args.round} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
                # model = load_checkpoint(os.path.join(save_dir, 'model.pt'), device=args.device, logger=logger)

                # test_preds = predict(
                #     model=model,
                #     data_loader=test_data_loader,
                #     scaler=scaler
                # )
                # test_scores, test_R2 = evaluate_predictions(
                #     preds=test_preds,
                #     targets=test_targets,
                #     num_tasks=args.num_tasks,
                #     metric_func=metric_func,
                #     dataset_type=args.dataset_type,
                #     logger=logger
                # )

            #     if len(test_preds) != 0:
            #         sum_test_preds += np.array(test_preds)
            #
            #     # Average test score
            #     avg_test_score = np.nanmean(test_scores)
            #     info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
            #     writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)
            #
            #     if args.show_individual_scores:
            #         # Individual test scores
            #         for task_name, test_score in zip(args.task_names, test_scores):
            #             info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
            #             writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)
            #
            # # Evaluate ensemble on test set
            # avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
            #
            # ensemble_scores, ensemble_R2 = evaluate_predictions(
            #     preds=avg_test_preds,
            #     targets=test_targets,
            #     num_tasks=args.num_tasks,
            #     metric_func=metric_func,
            #     dataset_type=args.dataset_type,
            #     logger=logger
            # )
            #
            # # Average ensemble score
            # avg_ensemble_test_score = np.nanmean(ensemble_scores)
            # info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
            # writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)
            #
            # # Individual ensemble scores
            # if args.show_individual_scores:
            #     for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            #         info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
            return train_loss, val_loss, best_epoch


    def iniatiate(self,train_data,val_data,args,logger):
        cache = False
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print

        if args.dataset_type == 'classification':
            class_sizes = get_class_sizes(train_data)
            debug('Class sizes')
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

        # if args.save_smiles_splits:
        #     save_smiles_splits(
        #         train_data=train_data,
        #         val_data=val_data,
        #         test_data=test_data,
        #         data_path=args.data_path,
        #         save_dir=args.save_dir
        #     )

        if args.features_scaling:
            # 对feature 进行normalize
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            # test_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        self.features_scaler=features_scaler
        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        if args.dataset_type == 'regression':
            debug('Fitting scaler')
            train_smiles, train_targets = train_data.smiles(), train_data.targets()
            scaler = StandardScaler().fit(train_targets)
            scaled_targets = scaler.transform(train_targets).tolist()
            train_data.set_targets(scaled_targets)
        else:
            scaler = None
        self.scaler=scaler
        # # Set up test set evaluation
        # test_smiles, test_targets = test_data.smiles(), test_data.targets()
        # if args.dataset_type == 'multiclass':
        #     sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        # else:
        #     sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))


        # Create data loaders
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache,
            # class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )

        train_show_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache,
        )

        val_data_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache
        )
        # test_data_loader = MoleculeDataLoader(
        #     dataset=test_data,
        #     batch_size=args.batch_size,
        #     num_workers=self.num_workers,
        #     cache=cache
        # )

        # # Tensorboard writer
        model_idx=0
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        self.save_dir=save_dir
        makedirs(save_dir)
        # try:
        #     writer = SummaryWriter(log_dir=save_dir)
        # except:
        #     writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if os.path.exists(os.path.join(save_dir, 'model.pt')):
            # debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0

        list_n_iter = []
        train_loss = []
        param_norm = []
        gradient_norm = []
        val_loss = []
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')
            n_iter, l_n_iter, a, b, c = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=self.loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                # writer=writer
            )

            list_n_iter.append(l_n_iter)
            train_loss.append(a)
            param_norm.append(b)
            gradient_norm.append(c)

            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores, val_R2 = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metric_func=self.metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            # evaluate train_set
            train_scores, train_R2 = evaluate(
                model=model,
                data_loader=train_show_data_loader,
                num_tasks=args.num_tasks,
                metric_func=self.metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=None
            )
            #
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            # writer.add_scalar(f'validation_{args.metric}', avg_val_score, epoch)
            # # 加入 val_loss
            # writer.add_scalar(f'validation_loss', np.nanmean(val_R2[1]), epoch)
            val_loss.append(np.nanmean(val_R2[1]))
            # # 加入 train_auc
            # writer.add_scalar(f'train_{args.metric}', np.nanmean(train_scores), epoch)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    # writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, epoch)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
                # 计算 best epoch里的train_auc
                best_train_score = np.nanmean(train_scores)

            # Evaluate on test set using model with best validation score
            info(f'Model {model_idx} best train {args.metric} = {best_train_score:.6f} on epoch {best_epoch}')
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            # model = load_checkpoint(os.path.join(save_dir, 'model.pt'), device=args.device, logger=logger)

            # test_preds = predict(
            #     model=model,
            #     data_loader=test_data_loader,
            #     scaler=scaler
            # )
            # test_scores, test_R2 = evaluate_predictions(
            #     preds=test_preds,
            #     targets=test_targets,
            #     num_tasks=args.num_tasks,
            #     metric_func=metric_func,
            #     dataset_type=args.dataset_type,
            #     logger=logger
            # )

        #     if len(test_preds) != 0:
        #         sum_test_preds += np.array(test_preds)
        #
        #     # Average test score
        #     avg_test_score = np.nanmean(test_scores)
        #     info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        #     writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)
        #
        #     if args.show_individual_scores:
        #         # Individual test scores
        #         for task_name, test_score in zip(args.task_names, test_scores):
        #             info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
        #             writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)
        #
        # # Evaluate ensemble on test set
        # avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
        #
        # ensemble_scores, ensemble_R2 = evaluate_predictions(
        #     preds=avg_test_preds,
        #     targets=test_targets,
        #     num_tasks=args.num_tasks,
        #     metric_func=metric_func,
        #     dataset_type=args.dataset_type,
        #     logger=logger
        # )
        #
        # # Average ensemble score
        # avg_ensemble_test_score = np.nanmean(ensemble_scores)
        # info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
        # writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)
        #
        # # Individual ensemble scores
        # if args.show_individual_scores:
        #     for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
        #         info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
        return train_loss,val_loss,best_epoch


    def predict_GNN(self,test_data,args,logger):
        cache = False
        if args.features_scaling:
            test_data.normalize_features(self.features_scaler)
        # Set up test set evaluation
        test_smiles, test_targets = test_data.smiles(), test_data.targets()
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        # Create data loaders
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache
        )
        model = load_checkpoint(os.path.join(self.save_dir, 'model.pt'), device=args.device, logger=logger)
        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=self.scaler
        )

        return test_preds

    def predict_GNN_last(self,test_data,iteration,args,logger):
        cache = False
        if args.features_scaling:
            test_data.normalize_features(self.features_scaler)
        # Set up test set evaluation
        test_smiles, test_targets = test_data.smiles(), test_data.targets()
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        # Create data loaders
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache
        )
        if args.round!=None:
            save_dir = os.path.join(self.save_dir_root, f'round_{args.round}')
        else:
            save_dir = os.path.join(self.save_dir_root, 'model_0')
        self.save_dir=save_dir
        try:
            model = load_checkpoint(os.path.join(self.save_dir, 'model_iteration_'+str(iteration+1)+'.pt'), device=args.device, logger=logger)
        except FileNotFoundError:
            model = load_checkpoint(os.path.join(self.save_dir, 'model.pt'), device=args.device, logger=logger)

        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=None
        )

        return test_preds

    def predict_GNN_last_tem(self,test_data,val_data,iteration,temperature,args,logger):
        cache = False
        if args.features_scaling:
            test_data.normalize_features(self.features_scaler)
        # Set up test set evaluation
        test_smiles, test_targets = test_data.smiles(), test_data.targets()
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        # Create data loaders
        val_data_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache
        )

        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=self.num_workers,
            cache=cache
        )
        if args.round!=None:
            save_dir = os.path.join(self.save_dir_root, f'round_{args.round}')
        else:
            save_dir = os.path.join(self.save_dir_root, 'model_0')
        self.save_dir=save_dir
        try:
            model = load_checkpoint(os.path.join(self.save_dir, 'model_iteration_'+str(iteration+1)+'.pt'), device=args.device, logger=logger)
        except FileNotFoundError:
            model = load_checkpoint(os.path.join(self.save_dir, 'model.pt'), device=args.device, logger=logger)


        tem = temperature_train(
            model=model,
            data_loader=val_data_loader,
            loss_func=self.loss_func,
            args=args)

        # self.temperature.append(tem)

        test_preds_tem = temperature_predict(
            model=model,
            data_loader=test_data_loader,
            scaler=None,
            temperature=tem[-1]
        )

        test_preds=predict(
            model=model,
            data_loader=test_data_loader,
            scaler=None
        )

        return test_preds_tem,test_preds,tem

    def evaluate_GNN(self,test_preds,test_data,args,logger,metric_list):

        results = evaluate_predictions_list(
            preds=test_preds,
            targets=test_data.targets(),
            num_tasks=args.num_tasks,
            metrics=metric_list,
            dataset_type=args.dataset_type,
            logger=logger
        )
        return results



    #     if len(test_preds) != 0:
    #         sum_test_preds += np.array(test_preds)
    #
    #     # Average test score
    #     avg_test_score = np.nanmean(test_scores)
    #     info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
    #     writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)
    #
    #     if args.show_individual_scores:
    #         # Individual test scores
    #         for task_name, test_score in zip(args.task_names, test_scores):
    #             info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
    #             writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)
    #
    # # Evaluate ensemble on test set
    # avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    #
    # ensemble_scores, ensemble_R2 = evaluate_predictions(
    #     preds=avg_test_preds,
    #     targets=test_targets,
    #     num_tasks=args.num_tasks,
    #     metric_func=metric_func,
    #     dataset_type=args.dataset_type,
    #     logger=logger
    # )
    #
    # # Average ensemble score
    # avg_ensemble_test_score = np.nanmean(ensemble_scores)
    # info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    # writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)
    #
    # # Individual ensemble scores
    # if args.show_individual_scores:
    #     for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
    #         info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
