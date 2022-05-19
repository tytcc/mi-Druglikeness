#!/Anaconda3/envs/ccjpython/python3
# Created on 2021/6/23 13:34
# Author:tyty
import torch.nn as nn

from chemprop.models.mpn import MPN
from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function, initialize_weights

class resMoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False,output=None):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e. outputting
                           learned features in the final layer before prediction.
        """
        super(resMoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        if output is not None:
            self.output_size=output
        else:
            self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

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
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                [dropout,
                nn.Linear(first_linear_dim, self.output_size)]
            ]
        else:
            ffn = [
                [dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)]
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.append([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.append([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        # self.ffn = nn.Sequential(*ffn)
        self.ffn=ffn
    def featurize(self, *input):
        """
        Computes feature vectors of the input by leaving out the last layer.
        :param input: Input.
        :return: The feature vectors computed by the MoleculeModel.
        """
        return self.ffn[:-1](self.encoder(*input))

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """
        if self.featurizer:
            return self.featurize(*input) #只计算graph model 计算出的encoder的编码结果

        output=self.encoder(*input)
        for i in range(len(self.ffn)):
            fn=nn.Sequential(*self.ffn[i])
            output = fn(output)+output

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output

class MultiConcatModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False,output=None):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e. outputting
                           learned features in the final layer before prediction.
        """
        super(MultiConcatModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'

        #sou
        self.sou_classification = args.sou_dataset_type == 'classification'
        self.sou_multiclass = args.sou_dataset_type == 'multiclass'

        self.featurizer = featurizer

        if output is not None:
            self.sou_output_size=output[0]
            self.tar_output_size=output[1]
        else:
            self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        #sou
        if self.sou_classification:
            self.sou_sigmoid = nn.Sigmoid()

        if self.sou_multiclass:
            self.sou_multiclass_softmax = nn.Softmax(dim=2)


        self.create_encoder(args)
        self.create_middle(args)
        self.create_target(args)
        self.create_source(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_middle(self, args: TrainArgs):
        """
        Creates the middle feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.middle_hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        ffn = [
            dropout,
            nn.Linear(first_linear_dim, args.ffn_hidden_size)
        ]

        # Create FFN model
        self.middle_ffn = nn.Sequential(*ffn)

    def create_source(self,args):
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        if args.sou_ffn_num_layers==1: # =1
            ffn=[
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.sou_output_size),
            ]
        elif args.sou_ffn_num_layers>=2:
            ffn=[
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.sou_ffn_hidden_size),
            ]
            for _ in range(args.sou_ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.sou_ffn_hidden_size, args.sou_ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.sou_ffn_hidden_size, self.sou_output_size),
            ])
        self.sou_ffn=nn.Sequential(*ffn)

    def create_target(self,args):
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        if args.tar_ffn_num_layers==1: # =1
            ffn=[
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.tar_output_size),
            ]
        elif args.tar_ffn_num_layers >= 2:
            ffn=[
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.tar_ffn_hidden_size),
            ]
            for _ in range(args.tar_ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.tar_ffn_hidden_size, args.tar_ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.tar_ffn_hidden_size, self.tar_output_size),
            ])
        self.tar_ffn=nn.Sequential(*ffn)

    def featurize(self, *input):
        """
        Computes feature vectors of the input by leaving out the last layer.
        :param input: Input.
        :return: The feature vectors computed by the MoleculeModel.
        """
        return self.ffn[:-1](self.encoder(*input))

    def forward(self,*input,sou=True):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """
        if self.featurizer:
            return self.featurize(*input) #只计算graph model 计算出的encoder的编码结果,也就是将其作为feature

        output = self.middle_ffn(self.encoder(*input))
        #sou
        if sou==True:
            sou_output = self.sou_ffn(output)
            if self.sou_classification and not self.training:
                sou_output = self.sou_sigmoid(sou_output)
            if self.sou_multiclass:
                sou_output = sou_output.reshape((sou_output.size(0), -1, self.num_classes)) #未改num_classes  # batch size x num targets x num classes per target
                if not self.training:
                    sou_output = self.multiclass_softmax(sou_output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
            return sou_output
        else:
            tar_output = self.tar_ffn(output)
            # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
            if self.classification and not self.training:
                tar_output = self.sigmoid(tar_output)
            if self.multiclass:
                tar_output = tar_output.reshape(
                    (tar_output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
                if not self.training:
                    tar_output = self.multiclass_softmax(
                        tar_output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
            return tar_output
