import json
import os
from tempfile import TemporaryDirectory
import pickle
from typing import List, Optional, Tuple
from typing_extensions import Literal

import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.features import get_available_features_generators


def get_checkpoint_paths(checkpoint_dir: Optional[str],
                         checkpoint_path: Optional[str], ext: str = '.pt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths.

    If checkpoint_dir is provided, walks the directory and collects all checkpoints.
    If checkpoint_path is provided, only collects that one checkpoint.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param checkpoint_path: Path to a checkpoint.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if both checkpoint_dir and checkpoint_path are None.
    """
    if checkpoint_dir is not None and checkpoint_path is not None:
        raise ValueError('Can only specify one of checkpoint_dir and checkpoint_path')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        return checkpoint_paths

    return None


class CommonArgs(Tap):
    """CommonArgs contains arguments that are used in both TrainArgs and PredictArgs."""

    smiles_column: str = None  # Name of the column containing SMILES strings. By default, uses the first column.
    checkpoint_dir: str = None  # Directory from which to load model checkpoints (walks directory and ensembles all models that are found)
    checkpoint_path: str = None  # Path to model checkpoint (.pt file)
    no_cuda: bool = False  # Turn off cuda (i.e. use CPU instead of GPU)
    gpu: int = None  # Which GPU to use
    features_generator: List[str] = None  # Method(s) of generating additional features
    features_path: List[str] = None  # Path(s) to features to use in FNN (instead of features_generator)
    sou_features_path: List[str] = None  # Path(s) to features to use in FNN (instead of features_generator)
    no_features_scaling: bool = False  # Turn off scaling of features
    no_sou_features_scaling: bool = False # Turn off scalinbg of features
    max_data_size: int = None  # Maximum number of data points to load
    num_workers: int = 8   # Number of workers for the parallel data loading (0 means sequential)
    batch_size: int = 50  # Batch size
    sou_batch_size: int=50

    ### new version chemprop
    phase_features_path: str = None
    atom_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    bond_features_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""
    no_cache_mol: bool = False
    """
    Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    """
    empty_cache: bool = False
    """
    Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._checkpoint_paths = None
        ##  new version chemprop
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0

    @property
    def device(self) -> torch.device:
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        return not self.no_features_scaling

    @property
    def sou_features_scaling(self) -> bool:
        return not self.no_sou_features_scaling

    @property
    def checkpoint_paths(self) -> List[str]:
        return self._checkpoint_paths

    @checkpoint_paths.setter
    def checkpoint_paths(self, checkpoint_paths: str) -> None:
        self._checkpoint_paths = checkpoint_paths

    def add_arguments(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_path=self.checkpoint_path
        )

class TrainArgs(CommonArgs):
    """TrainArgs includes CommonArgs along with additional arguments used for training a chemprop model."""

    # General arguments
    # sou_data_path: str = None #source data_path
    round: int = None
    data_path: str = None  # Path to data CSV file
    sou_data_path: str = None
    target_columns: List[str] = None  # Name of the columns containing target values. By default, uses all columns except the SMILES column.
    sou_dataset_type:Literal['regression', 'classification', 'multiclass'] = 'classification'
    dataset_type: Literal['regression', 'classification', 'multiclass'] = 'classification'  # Type of dataset. This determines the loss function used during training.
    multiclass_num_classes: int = 3  # Number of classes when running multiclass classification
    separate_val_path: str = None  # Path to separate val set, optional
    separate_test_path: str = None  # Path to separate test set, optional
    split_type: Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'] = 'random'  # Method of splitting the data into train/val/test
    split_sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # Split proportions for train/validation/test sets
    num_folds: int = 1  # Number of folds when performing cross validation
    folds_file: str = None  # Optional file of fold labels
    val_fold_index: int = None  # Which fold to use as val for leave-one-out cross val
    test_fold_index: int = None  # Which fold to use as test for leave-one-out cross val
    crossval_index_dir: str = None  # Directory in which to find cross validation index files
    crossval_index_file: str = None  # Indices of files to use as train/val/test. Overrides --num_folds and --seed.
    sou_num_tasks: int = None
    seed: int = 33  # Random seed to use when splitting data into train/val/test sets. When `num_folds` > 1, the first fold uses this seed and all subsequent folds add 1 to the seed.
    pytorch_seed: int = 3  # Seed for PyTorch randomness (e.g. random initial weights)
    metric: Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy','mcc','f1','gmeans','precision','recall'] = None  # Metric to use during evaluation. Defaults to "auc" for classification and "rmse" for regression.
    save_dir: str = None  # Directory where model checkpoints will be saved
    save_smiles_splits: bool = False  # Save smiles for each train/val/test splits for prediction convenience later
    test: bool = False  # Whether to skip training and only test the model
    quiet: bool = False  # Skip non-essential print statements
    log_frequency: int = 1  # The number of batches between each logging of the training loss
    show_individual_scores: bool = False  # Show all scores for individual targets, not just average, at the end
    cache_cutoff: int = 10000  # Maximum number of molecules in dataset to allow caching. Below this number, caching is used and data loading is sequential. Above this number, caching is not used and data loading is parallel.
    fold: int=None

    ## newversion chemprop
    loss_function: Literal['mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet'] = 'binary_cross_entropy'
    """Choice of loss function. Loss functions are limited to compatible dataset types."""

    # Model arguments
    bias: bool = False  # Whether to add bias to linear layers
    hidden_size: int = 300  # Dimensionality of hidden layers in MPN
    depth: int = 3  # Number of message passing steps
    dropout: float = 0.0  # Dropout probability
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'  # Activation function
    atom_messages: bool = False  # Centers messages on atoms instead of on bonds
    undirected: bool = False  # Undirected edges (always sum the two relevant bond vectors)
    undirected: bool = False  # Undirected edges (always sum the two relevant bond vectors)
    ffn_hidden_size: int = None  # Hidden dim for higher-capacity FFN (defaults to hidden_size)
    ffn_num_layers: int = 2  # Number of layers in FFN after MPN encoding
    features_only: bool = False  # Use only the additional features in an FFN, no graph network
    separate_val_features_path: List[str] = None  # Path to file with features for separate val set
    separate_test_features_path: List[str] = None  # Path to file with features for separate test set
    sou_separate_test_features_path: List[str] = None # Path to file with features for separate test set
    config_path: str = None  # Path to a .json file containing arguments. Any arguments present in the config file will override arguments specified via the command line or by the defaults.
    ensemble_size: int = 1  # Number of models in ensemble

    ## outputfile
    outputfile:str=None # inference file name 

    # wang previous args

    s: str = None # output path
    m: str=None # load model path
    mode: str = None # active , passive ,normal
    weight_decay: float=0

    #new_version 1.3.0
    ignore_columns: List[str] = None
    data_weights_path : str = None
    separate_val_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_bond_features_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_bond_features_path: str = None
    """Path to file with extra atom descriptors for separate test set."""

    overwrite_default_atom_features: bool = False
    """
    Overwrites the default atom descriptors with the new ones instead of concatenating them.
    Can only be used if atom_descriptors are used as a feature.
    """
    no_atom_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    overwrite_default_bond_features: bool = False
    """Overwrites the default atom descriptors with the new ones instead of concatenating them"""
    no_bond_features_scaling: bool = False
    """Turn off atom feature scaling."""

    # Training arguments
    epochs: int = 50  # Number of epochs to run
    warmup_epochs: float = 2.0  # Number of epochs during which learning rate increases linearly from init_lr to max_lr. Afterwards, learning rate decreases exponentially from max_lr to final_lr.
    init_lr: float = 1e-4  # Initial learning rate
    max_lr: float = 1e-3  # Maximum learning rate
    final_lr: float = 1e-4  # Final learning rate
    class_balance: bool = False  # Trains with an equal number of positives and negatives in each batch (only for single task classification)

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._checkpoint_paths = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def minimize_score(self) -> bool:
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy'}

    @property
    def use_input_features(self) -> bool:
        return self.features_generator is not None or self.features_path is not None

    @property
    def num_lrs(self) -> int:
        return 1  # Number of learning rates

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    @num_tasks.setter
    def num_tasks(self, num_tasks: int) -> None:
        self._num_tasks = num_tasks

    @property
    def features_size(self) -> int:
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    def add_arguments(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        global temp_dir  # Prevents the temporary directory from being deleted upon function return

        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            temp_dir = TemporaryDirectory()
            self.save_dir = temp_dir.name

        # Fix ensemble size if loading checkpoints
        if self.checkpoint_paths is not None and len(self.checkpoint_paths) > 0:
            self.ensemble_size = len(self.checkpoint_paths)

        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'auc'
            elif self.dataset_type == 'multiclass':
                self.metric = 'cross_entropy'
            else:
                self.metric = 'rmse'

        if not ((self.dataset_type == 'classification' and self.metric in ['auc', 'prc-auc', 'accuracy','mcc','f1','gmeans','precision','recall']) or
                (self.dataset_type == 'regression' and self.metric in ['rmse', 'mae', 'mse', 'r2']) or
                (self.dataset_type == 'multiclass' and self.metric in ['cross_entropy', 'accuracy'])):
            raise ValueError(f'Metric "{self.metric}" invalid for dataset type "{self.dataset_type}".')

        # Validate class balance
        if self.class_balance and self.dataset_type != 'classification':
            raise ValueError('Class balance can only be applied if the dataset type is classification.')

        # Validate features
        if self.features_only and not (self.features_generator or self.features_path):
            raise ValueError('When using features_only, a features_generator or features_path must be provided.')

        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator and self.features_scaling:
            raise ValueError('When using rdkit_2d_normalized features, --no_features_scaling must be specified.')

        # Handle FFN hidden size
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        # Handle MPN variants
        if self.atom_messages and self.undirected:
            raise ValueError('Undirected is unnecessary when using atom_messages '
                             'since atom_messages are by their nature undirected.')

        # Validate split type settings
        if not (self.split_type == 'predetermined') == (self.folds_file is not None) == (self.test_fold_index is not None):
            raise ValueError('When using predetermined split type, must provide folds_file and test_fold_index.')

        if not (self.split_type == 'crossval') == (self.crossval_index_dir is not None):
            raise ValueError('When using crossval split type, must provide crossval_index_dir.')

        if not (self.split_type in ['crossval', 'index_predetermined']) == (self.crossval_index_file is not None):
            raise ValueError('When using crossval or index_predetermined split type, must provide crossval_index_file.')

        if self.split_type in ['crossval', 'index_predetermined']:
            with open(self.crossval_index_file, 'rb') as rf:
                self._crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            self.seed = 0

        # Test settings
        if self.test:
            self.epochs = 0


class PredictArgs(CommonArgs):
    """PredictArgs includes CommonArgs along with additional arguments used for predicting with a chemprop model."""

    test_path: str  # Path to CSV file containing testing data for which predictions will be made
    preds_path: str  # Path to CSV file where predictions will be saved

    @property
    def ensemble_size(self) -> int:
        return len(self._checkpoint_paths)


class HyperoptArgs(TrainArgs):
    """HyperoptArgs includes TrainArgs along with additional arguments used for optimizing chemprop hyperparameters."""

    num_iters: int = 20  # Number of hyperparameter choices to try
    config_save_path: str  # Path to .json file where best hyperparameter settings will be written
    log_dir: str = None  # (Optional) Path to a directory where all results of the hyperparameter optimization will be written


class SklearnTrainArgs(TrainArgs):
    """SklearnTrainArgs includes TrainArgs along with additional arguments for training a scikit-learn model."""

    model_type: Literal['random_forest', 'svm']  # scikit-learn model to use
    class_weight: Literal['balanced'] = None  # How to weight classes (None means no class balance)
    single_task: bool = False  # Whether to run each task separately (needed when dataset has null entries)
    radius: int = 2  # Morgan fingerprint radius
    num_bits: int = 2048  # Number of bits in morgan fingerprint
    num_trees: int = 500  # Number of random forest trees


class SklearnPredictArgs(Tap):
    """SklearnPredictArgs contains arguments used for predicting with a trained scikit-learn model."""

    test_path: str  # Path to CSV file containing testing data for which predictions will be made
    smiles_column: str = None  # Name of the column containing SMILES strings. By default, uses the first column.
    preds_path: str  # Path to CSV file where predictions will be saved
    dataset_type: Literal['classification', 'regression']  # Type of dataset
    model_type: Literal['random_forest', 'svm']  # scikit-learn model to use
    checkpoint_dir: str = None  # Path to directory containing model checkpoints (.pkl file)
    checkpoint_path: str = None  # Path to model checkpoint (.pkl file)
    radius: int = 2  # Morgan fingerprint radius
    num_bits: int = 2048  # Number of bits in morgan fingerprint
    num_tasks: int  # Number of tasks the trained model makes predictions for

    def __init__(self, *args, **kwargs) -> None:
        super(SklearnPredictArgs, self).__init__(*args, **kwargs)
        self._checkpoint_paths = None

    @property
    def checkpoint_paths(self) -> List[str]:
        return self._checkpoint_paths

    @checkpoint_paths.setter
    def checkpoint_paths(self, checkpoint_paths: str) -> None:
        self._checkpoint_paths = checkpoint_paths

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_path=self.checkpoint_path,
            ext='.pkl'
        )
