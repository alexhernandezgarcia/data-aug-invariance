"""
Utils functions for training deep artificial neural networks
"""
import os
import sys
import re
import shutil
import yaml
import csv
import time
import numpy as np
import dask.array as da
import h5py
import pickle
from argparse import Namespace
from functools import partial, update_wrapper

from data_input import validation_image_params, create_control_dataset
from data_input import get_generator, batch_generator
from activations import get_activations

import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.layers import Layer
from tensorflow.compat.v1.keras.callbacks import Callback
from tensorflow.compat.v1.keras.losses import categorical_crossentropy
import tensorflow.compat.v1 as tf

class TrainingProgressLogger():

    def __init__(self, output_file, model, config, images, labels, layers=[], 
                 every_iter=None, every_epoch=None, **kwargs):
        self.output_file = open(output_file, 'w')
        self.model = model
        if layers:
            self.layers = layers
        else:
            self.layers = config.logger.layers
        self.layers_dim = []
        if every_iter:
            self.every_iter = every_iter
        else:
            self.every_iter = config.logger.every_iter
        if every_epoch:
            self.every_epoch = every_epoch
        else:
            self.every_epoch = config.logger.every_epoch
        self.images = images
        self.labels = labels
        self.activation_functions = []
        self.activations = []

        # Create log dataset
        daug_params_file = get_daug_scheme_path(config.logger.daug, 
                                                config.data.data_file)
        daug_params = yaml.load(open(daug_params_file, 'r'),
                                Loader=yaml.FullLoader)
        nodaug_params_file = get_daug_scheme_path(config.logger.daug, 
                                                config.data.data_file)
        self.nodaug_params = validation_image_params(config.daug.nodaug, 
                                                **daug_params)
        self.images, self.labels = create_control_dataset(
                images, labels, daug_params, self.nodaug_params,
                n_per_image=config.logger.n_per_image, 
                n_per_class=config.logger.n_per_class, 
                chunk_size=config.logger.chunk_size,
                seed=config.logger.seed)
        self.n_images = self.images.shape[0]

        # Setup data generator
        image_gen = get_generator(images, **self.nodaug_params)
        self.batch_gen = batch_generator(image_gen, self.images, self.labels, 
                                         config.logger.batch_size, 
                                         aug_per_im=1, shuffle=False)
        # Determine activation functions
        for layer_name in self.layers:
            layer = model.get_layer(layer_name)
            self.layers_dim.append(np.prod(layer.output_shape[1:]))
            self.activation_functions.append(
                    K.function([model.input, K.learning_phase()], 
                               [layer.output]))

        # setup CSV writer
        self.writer = csv.writer(self.output_file, delimiter=',',
                                 quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write headers
        headers = model.metrics_names[:]
        for layer_name, layer_dim in zip(self.layers, self.layers_dim):
            layer_header = ['{}_dim{:05d}_data{:05d}'.format(
                layer_name, dim, x) for x in range(self.n_images) 
                for dim in range(layer_dim)]
            headers.extend(layer_header)
        self.writer.writerow(headers)

    def log(self, metrics):
        row = list(list(zip(*metrics))[1])
        for activations in self.activations:
            row.extend(activations)
        self.writer.writerow(row)
        self.activations = []

    def get_activations(self):
        for activation_function in self.activation_functions:
            activations = get_activations(activation_function, self.batch_gen)
            self.activations.append(
                    da.ravel(da.squeeze(activations)).compute().tolist())


    def close(self):
        self.output_file.close()



def pairwise_loss(y_true, y_pred):
    """
    Computes the relative loss (MSE) of a set of relevant pairs of samples,
    determined by the mask y_true.

    Parameters
    ----------
    y_true : ndarray
        Boolean mask to determine the relevant pairs of samples. For image
        identification, these are the samples that belong to the same source
        image.

    y_pred : ndarray
        Squared matrix of pairwise losses, typically the mean squared error
        (MSE) for image identification. 

    Returns
    -------
        Relative mean loss of the relevant pairs, that is the mean loss of the 
        relevant pairs normalized by the mean loss of all pairs. 
    """
    return K.sum(y_true * y_pred) / K.sum(y_true) / (K.mean(y_pred) + K.epsilon())


def daug_inv_loss(y_true, y_pred):
    """
    Computes the relative loss (MSE) of a set of relevant pairs of samples,
    determined by the mask y_true.

    Parameters
    ----------
    y_true : ndarray
        Boolean mask to determine the relevant pairs of samples. For image
        identification, these are the samples that belong to the same source
        image.

    y_pred : ndarray
        Squared matrix of pairwise losses, typically the mean squared error
        (MSE) for image identification. 

    Returns
    -------
        Relative mean loss of the relevant pairs, that is the mean loss of the 
        relevant pairs normalized by the mean loss of all pairs. 
    """
    y_pred = y_pred[:, :, 0]
    y_daug = y_true[:, :, 0]
    y_class = y_true[:, :, 1]
    return (K.sum(y_daug * y_pred) / K.sum(y_daug)) / \
           (K.sum(y_class * y_pred) / K.sum(y_class))


def invariance_loss(y_true, y_pred):
    """
    Computes the relative loss (MSE) of a set of relevant pairs of samples,
    determined by the mask y_true.

    Parameters
    ----------
    y_true : ndarray
        Boolean mask to determine the relevant pairs of samples. For image
        identification, these are the samples that belong to the same source
        image.

    y_pred : ndarray
        Squared matrix of pairwise losses, typically the mean squared error
        (MSE) for image identification. 

    Returns
    -------
        Relative mean loss of the relevant pairs, that is the mean loss of the 
        relevant pairs normalized by the mean loss of all pairs. 
    """
    y_pred = y_pred[:, :, 0]
    y_rel = y_true[:, :, 0]
    y_all = y_true[:, :, 1]
    return (K.sum(y_rel * y_pred) / K.sum(y_rel)) / \
           (K.sum(y_all * y_pred) / K.sum(y_all))


def weighted_loss(loss_function, weight):
    if loss_function == 'categorical_crossentropy':
        loss_function = categorical_crossentropy
    def loss(y_true, y_pred):
        return loss_function(y_true, y_pred) * weight
    return loss


def mean_loss_mod(y_true, y_pred):
    """
    Simply returns the mean of y_pred across the batch, regardless of y_true.

    Parameters
    ----------
    y_true : ndarray
        Irrelevant.

    y_pred : ndarray
        Irrelevant

    Returns
    -------
        The mean of y_pred
    """
    y_true_mod = y_true
    y_true_mod[:, :5]

    return K.mean(y_true_mod)


def mean_loss(y_true, y_pred):
    """
    Simply returns the mean of y_pred across the batch, regardless of y_true.

    Parameters
    ----------
    y_true : ndarray
        Irrelevant.

    y_pred : ndarray
        Irrelevant

    Returns
    -------
        The mean of y_pred
    """
    return K.mean(y_pred)


def sum_inv_losses(losses, loss_weights):
    def inv_sum(y_true, y_pred):
        mses = []
        for loss, weight in zip(losses.items(), loss_weights.items()):
            if ('inv' in loss[0]) & (loss[0] == weight[0]):
               mses.append(weight[1] * loss[1](y_true, y_pred))
        return K.sum(mses)
    return inv_sum


def handle_train_dir(train_dir):
    """
    Checks if the training directory exists and creates it or overwrites it
    according to the user input.

    Parameters
    ----------
    train_dir : str
        Training directory, where the checkpoint and other files are stored

    """
    if os.path.exists(train_dir):
        print('Train directory %s already exists!' % train_dir)
        if _confirm('Do you want to overwrite its contents? [Y/N]: '):
            shutil.rmtree(train_dir)
            os.makedirs(train_dir)
        else:
            if _confirm('Do you want to create the new files for alongside the'
                        'old ones? [Y/N]: '):
                pass
            else:
                raise ValueError('Cannot establish a train directory')
    else:
        os.makedirs(train_dir)


def prepare_train_config(train_config, flags):
    """
    Modifies the input train configuration file according to the arguments of
    the current execution. The specifications passed as arguments prevail over
    the settings from the configuration file. Not every single field can be
    modified from the flags.

    Parameters
    ----------
    train_config: dict
        The train configuration dictionary, read from a YAML file

    flags : argparse.Namespace
        The flag aguments of the execution

    Return
    ------
    train_config : dict
        The updated configuration
    """

    # Get data set
    if flags.data_file is None:
        flags.data_file = train_config['data']['data_file']

    if 'cifar' in flags.data_file:
        dataset = 'cifar'
    elif 'mnist_rot' in flags.data_file:
        dataset = 'mnist_rot'
    elif 'mnist' in flags.data_file:
        dataset = 'mnist'
    elif 'tinyimagenet' in flags.data_file:
        dataset = 'tinyimagenet'
    elif 'imagenet' in flags.data_file:
        dataset = 'imagenet'
    elif 'emotion' in flags.data_file:
        dataset = 'emotion'
    elif 'catsndogs' in flags.data_file:
        dataset = 'catsndogs'
    elif 'synthetic' in flags.data_file:
        dataset = 'synthetic'
    elif 'niko92' in flags.data_file:
        dataset = 'niko92'
    elif 'kamitani1200' in flags.data_file:
        dataset = 'kamitani1200'
    else:
        raise NotImplementedError()

    # Set data configuration
    if 'data' not in train_config:
        train_config.update({'data': {
            'data_file': flags.data_file,
            'group_tr': flags.group_tr,
            'group_val': flags.group_val,
            'labels_id': flags.labels_id,
            'pct_tr': flags.pct_tr,
            'pct_val': flags.pct_val,
            'shuffle_train_val': flags.shuffle_train_val,
            'chunk_size': flags.chunk_size}})
    if flags.data_file:
        train_config['data']['data_file'] = flags.data_file
    if flags.group_tr:
        train_config['data']['group_tr'] = flags.group_tr
    if flags.group_val:
        train_config['data']['group_val'] = flags.group_val
    if flags.labels_id:
        train_config['data']['labels_id'] = flags.labels_id
    if flags.pct_tr:
        train_config['data']['pct_tr'] = flags.pct_tr
    if flags.pct_val:
        train_config['data']['pct_val'] = flags.pct_val
    if flags.shuffle_train_val:
        train_config['data']['shuffle_train_val'] = flags.shuffle_train_val

    if train_config['data']['group_val']:
        train_config['data']['pct_val'] = None

    # Set data augmentaiton configuration
    if 'daug' not in train_config:
        train_config.update({'daug': {
            'daug_params_file': os.path.join(
                'daug_schemes', dataset, flags.daug_params),
            'aug_per_img_tr': flags.aug_per_img_tr,
            'aug_per_img_val': flags.aug_per_img_val}})
    train_config['daug'].update({'nodaug': os.path.join(
        'daug_schemes', dataset, 'nodaug.yml')})
    if flags.daug_params:
        train_config['daug']['daug_params_file'] = os.path.join(
                'daug_schemes', dataset, flags.daug_params)
    else:
        train_config['daug']['daug_params_file'] = os.path.join(
                'daug_schemes', dataset, 
                train_config['daug']['daug_params_file'])
    if flags.aug_per_img_tr:
        train_config['daug']['aug_per_img_tr'] = flags.aug_per_img_tr
    if flags.aug_per_img_val:
        train_config['daug']['aug_per_img_val'] = flags.aug_per_img_val

    # Set seed configuration
    if 'seeds' not in train_config:
        train_config.update({'seeds': {
            'tf': flags.seed_tf,
            'np': flags.seed_np,
            'daug': flags.seed_daug,
            'batch_shuffle': flags.seed_batch_shuffle,
            'train_val': flags.seed_train_val}})
    if flags.seed_tf:
        train_config['seeds']['tf'] = flags.seed_tf
    if flags.seed_np:
        train_config['seeds']['np'] = flags.seed_np
    if flags.seed_daug:
        train_config['seeds']['daug'] = flags.seed_daug
    if flags.seed_batch_shuffle:
        train_config['seeds']['batch_shuffle'] = flags.seed_batch_shuffle
    if flags.seed_train_val:
        train_config['seeds']['train_val'] = flags.seed_train_val

    # Set metrics
    if flags.metrics:
        train_config['metrics'] = flags.metrics

    # Set train hyperparameters
    if flags.epochs is not None:
        train_config['train']['epochs'] = flags.epochs
    if flags.batch_size:
        train_config['train']['batch_size']['tr'] = flags.batch_size
    if flags.learning_rate:
        train_config['train']['lr']['init_lr'] = flags.learning_rate

    # Set invariance parameters files
    if flags.force_invariance:
        train_config['optimizer']['invariance'] = True
    else:
        train_config['optimizer']['invariance'] = False
    if flags.daug_invariance_params:
        train_config['optimizer']['daug_invariance_params_file'] = \
                os.path.join('invariance_params', flags.daug_invariance_params)
    elif 'daug_invariance_params_file' in train_config['optimizer']:
        train_config['optimizer']['daug_invariance_params_file'] = \
                os.path.join(
                    'invariance_params', 
                    train_config['optimizer']['daug_invariance_params_file'])
    else:
        pass
    if flags.class_invariance_params:
        train_config['optimizer']['class_invariance_params_file'] = \
                os.path.join('invariance_params', 
                             flags.class_invariance_params)
    elif 'class_invariance_params_file' in train_config['optimizer']:
        train_config['optimizer']['class_invariance_params_file'] = \
                os.path.join(
                    'invariance_params', 
                    train_config['optimizer']['class_invariance_params_file'])
    else:
        pass

    # Set network parameters
    if flags.network_name:
        train_config['network']['name'] = flags.network_name
    if flags.weight_decay:
        train_config['network']['reg']['weight_decay'] = flags.weight_decay
    if flags.dropout:
        if isinstance(flags.dropout, bool):
            train_config['network']['reg']['dropout'] = True
        elif isinstance(flags.dropout, float):
            train_config['network']['reg']['dropout'] = flags.dropout
        else:
            pass
    if flags.no_dropout:
        train_config['network']['reg']['dropout'] = False
    if flags.batch_norm:
        train_config['network']['batch_norm'] = True
    if flags.no_batch_norm:
        train_config['network']['batch_norm'] = False

    # Set simulation parameters
    if flags.simulate_norep_samples:
        train_config['train']['simulate']['norep_samples'] = True
    if flags.simulate_rep_samples:
        train_config['train']['simulate']['rep_samples'] = \
                flags.simulate_rep_samples
    if flags.simulate_bs_lr:
        train_config['train']['simulate']['bs_lr'] = flags.simulate_bs_lr

    return train_config


def prepare_test_config(test_config, flags):
    """
    Prepares a generic test configuration dictionary for use and modifies the
    configuration according to the arguments passed as flags.

    Parameters
    ----------
    test_config: dict
        The test configuration dictionary, read from a YAML file

    flags : argparse.Namespace
        The flag aguments of the execution

    Return
    ------
    test_config : dict
        The updated configuration
    """

    def _add_recursively(config_dict, metrics, dataset, root='daug_schemes'):
        """
        Recursively adds or updates value of the configuration dictionary:
            - metrics
            - Read the data augmentation configuration dictionaries

        Parameters
        ----------
        config_dict : dict

        metrics : list str
            The list of metrics to compute

        dataset : str
            The name of the data set, which must be the name of a folder within
            root.

        root : str
            The root directory where the data augmentation configuration files
            are stored. It must contain a folder named dataset.

        Return
        ------
        config_dict : dict
            The modified dictionary
        """
        for k, v in config_dict.items():
            if 'daug_params' in k:
                filename = os.path.join(root, dataset, v)
                with open(filename, 'r') as yml_file:
                    config_dict[k]= yaml.load(yml_file, Loader=yaml.FullLoader)
            elif metrics is not None and k == 'metrics':
                config_dict[k] = metrics
            elif isinstance(v, dict):
                _add_recursively(v, metrics, dataset, root)
            else:
                pass

        return config_dict

    # Get data set
    if 'cifar' in flags.data_file:
        dataset = 'cifar'
    elif 'mnist_rot' in flags.data_file:
        dataset = 'mnist_rot'
    elif 'mnist' in flags.data_file:
        dataset = 'mnist'
    elif 'tinyimagenet' in flags.data_file:
        dataset = 'tinyimagenet'
    elif 'imagenet' in flags.data_file:
        dataset = 'imagenet'
    elif 'emotion' in flags.data_file:
        dataset = 'emotion'
    elif 'catsndogs' in flags.data_file:
        dataset = 'catsndogs'
    elif 'synthetic' in flags.data_file:
        dataset = 'synthetic'
    else:
        raise NotImplementedError()

    # Add performance on original images
    if 'test' in test_config and \
       'orig' not in test_config['test'] and \
       hasattr(flags, 'orig') and \
       flags.orig == True:
           test_config['test'].update({'orig': {'daug_params': 'nodaug.yml',
                                                'metrics': flags.metrics}})

    # Add test data augmentation performance
    if hasattr(flags, 'daug_schemes') and flags.daug_schemes:
        if 'test' in test_config and \
           'daug' not in test_config['test']:
            test_config['test'].update({'daug': {}})
        for scheme in flags.daug_schemes:
            if scheme not in test_config['test']['daug']:
                test_config['test']['daug'].update(
                        {scheme: {'daug_params': '{}.yml'.format(scheme),
                                  'repetitions': flags.daug_rep,
                                  'metrics': flags.metrics}})

    # Edit adversarial robustness configuration
    if hasattr(flags, 'adv_pct_data') and flags.adv_pct_data and \
            flags.adv_pct_data < 1.:
        adv_shuffle = True
        adv_shuffle_seed = flags.adv_shuffle_seed
    else:
        adv_shuffle = False
        adv_shuffle_seed = 0

    if hasattr(flags, 'adv_model') and flags.adv_model:
        if 'adv' not in test_config:
            test_config.update({'adv': 
                {'attacks': flags.adv_attacks,
                 'daug_params': 'nodaug.yml',
                 'black_box_model': flags.adv_model,
                 'pct_data': flags.adv_pct_data,
                 'shuffle_data': adv_shuffle,
                 'shuffle_seed': adv_shuffle_seed}})
        else:
            test_config['adv'].update(
                    {'black_box_model': flags.adv_model})

    if hasattr(flags, 'adv_attacks') and flags.adv_attacks:
        if 'adv' not in test_config:
            test_config.update({'adv': 
                {'attacks': flags.adv_attacks,
                 'daug_params': 'nodaug.yml',
                 'black_box_model': flags.adv_model,
                 'pct_data': flags.adv_pct_data,
                 'shuffle_data': adv_shuffle,
                 'shuffle_seed': adv_shuffle_seed}})
        else:
            test_config['adv'].update({'attacks': flags.adv_attacks})
        
    # Read attack configuration files
    if 'adv' in test_config:
        flags.adv_attacks = test_config['adv']['attacks']
        del test_config['adv']['attacks']
        test_config['adv'].update({'attacks': {}})
        for attack_file in flags.adv_attacks:
            with open(attack_file, 'r') as yml_file:
                attack = yaml.load(yml_file, Loader=yaml.FullLoader)
                test_config['adv']['attacks'].update({attack_file: attack})

        # Set percentage of data to compute adversarial robustness
        if hasattr(flags, 'adv_pct_data') and flags.adv_pct_data:
            test_config['adv'].update({'pct_data': flags.adv_pct_data})

    # Edit ablation configuration
    if hasattr(flags, 'ablation_pct') and flags.ablation_pct:
        if 'ablation' not in test_config:
            test_config.update({'ablation': {'pct': flags.ablation_pct,
                                             'daug_params': 'nodaug.yml',
                                             'metrics': flags.metrics,
                                             'repetitions': 1,
                                             'seed': 19}})
        else:
            test_config['ablation'].update({'pct': flags.ablation_pct})

    if hasattr(flags, 'ablation_rep') and flags.ablation_rep:
        if flags.ablation_rep > 1:
            flags.ablation_seed = None
        if 'ablation' not in test_config:
            test_config.update({'ablation': {'pct': [0.5],
                                             'layer_regexflags.': 
                                             layer_regex_ablation,
                                             'daug_params': 'nodaug.yml',
                                             'metrics': flags.metrics,
                                             'repetitions': flags.ablation_rep,
                                             'seed': flags.ablation_seed}})
        else:
            test_config['ablation'].update({'repetitions': flags.ablation_rep,
                                            'seed': flags.ablation_seed})

    if hasattr(flags, 'metrics') and len(flags.metrics) > 0:
        metrics = flags.metrics
    else:
        metrics = None

    # Edit daug_params values
    test_config = _add_recursively(test_config, metrics, dataset)

    return test_config  


def get_daug_scheme_path(daug_scheme, data_file, root='./'):

    # Get data set
    if 'cifar' in data_file:
        dataset = 'cifar'
    elif 'mnist_rot' in data_file:
        dataset = 'mnist_rot'
    elif 'mnist' in data_file:
        dataset = 'mnist'
    elif 'tinyimagenet' in data_file:
        dataset = 'tinyimagenet'
    elif 'imagenet' in data_file:
        dataset = 'imagenet'
    elif 'synthetic' in data_file:
        dataset = 'synthetic'
    elif 'niko92' in data_file:
        dataset = 'niko92'
    elif 'kamitani1200' in data_file:
        dataset = 'kamitani1200'
    else:
        raise NotImplementedError()

    daug_scheme_path = os.path.join(root, 'daug_schemes', dataset, 
                                    daug_scheme)

    return daug_scheme_path



def handle_metrics(metrics):
    metric_funcs = []
    metric_names = []
    for metric in metrics:
        if metric.lower() in ('accuracy', 'acc', 'categorical_accuracy'):
            from tensorflow.compat.v1.keras.metrics import categorical_accuracy
            metric_funcs.append(categorical_accuracy)
            metric_names.append('acc')
        elif metric.lower() in ('mae', 'mean_absolute_error'):
            from tensorflow.compat.v1.keras.losses import mean_absolute_error
            metric_funcs.append(mean_absolute_error)
            metric_names.append('mae')
        elif metric.lower().startswith('top'):
            k = int(metric[-1])
            from tensorflow.compat.v1.keras.metrics import top_k_categorical_accuracy
            metric_funcs.append(update_wrapper(
                partial(top_k_categorical_accuracy, k=k), 
                        top_k_categorical_accuracy))
            metric_names.append('top{:d}'.format(k))
        else:
            raise NotImplementedError()

    return metric_funcs, metric_names


def change_metrics_names(model, invariance):
    model.metrics_names = [s.replace('softmax', 'cat') for s 
            in model.metrics_names]
    model.metrics_names = [s.replace('categorical_accuracy', 'acc') for s 
            in model.metrics_names]
#     model.metrics_names = [s.replace('top_k', 'top{}'.format(m.keywords['k'])) if hasattr(m, 'keywords') else s for s, m in zip(model.metrics_names, model.metrics)]

    if not invariance:
        model.metrics_names = ['cat_{}'.format(s) for s in model.metrics_names]

    return model


def sel_metrics(metrics_names, metrics, no_mean, no_val_daug=False, 
                metrics_cat=[]):
    metrics_progbar = list(zip(metrics_names, metrics))
    for metric_name, metric in zip(metrics_names, metrics):
        if no_mean and 'mean_' in metric_name:
            metrics_progbar.remove((metric_name, metric))
        elif no_val_daug and 'val_daug_' in metric_name:
            metrics_progbar.remove((metric_name, metric))
        elif metric_name in metrics_cat:
            metrics_progbar.remove((metric_name, metric))
        else:
            pass

    return metrics_progbar


def dict2namespace(data_dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace

def namespace2dict(data_namespace):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    data_dict = {}
    for k in vars(data_namespace):
        if isinstance(getattr(data_namespace, k), Namespace):
            data_dict.update({k: namespace2dict(getattr(data_namespace, k))})
        else:
            data_dict.update({k: getattr(data_namespace, k)})

    return data_dict



def numpy_to_python(results_dict):
    """
    Recursively converts the numpy types into native Python types in order to
    enable proper dumping into YAML files:

    Parameters
    ----------
    results_dict : dict
        The input dictionary

    Return
    ------
    results_dict : dict
        The modified dictionary
    """
    def convert(v):
        if isinstance(v, np.ndarray):
            if np.ndim(v) == 1:
                return v.tolist()
        elif isinstance(v, (int, np.integer)):
            return int(v)
        elif isinstance(v, (float, np.float, np.float32)):
            return float(v)
        elif isinstance(v, list):
            for idx, el in enumerate(v):
                v[idx] = convert(el)
            return v
        elif isinstance(v, dict):
            return numpy_to_python(v)
        elif isinstance(v, Namespace):
            return numpy_to_python(vars(v))
        else:
            return v

    for k, v in results_dict.items():
        if isinstance(v, dict):
            numpy_to_python(v)
        elif isinstance(v, Namespace):
            numpy_to_python(vars(v))
        else:
            results_dict[k] = convert(v)

    return results_dict


def _confirm(message):
    """
    Ask user to enter Y or N (case-insensitive)

    Parameters
    ----------
    message : str
        Message to ask for confirmation

    Returns
    -------
    answer : bool
        True if the answer is Y.
    """
    answer = ""
    while answer not in ["y", "n"]:
        try: # Python 2.x
            answer = raw_input(message).lower()
        except NameError: # Python 3.x
            answer = input(message).lower()
    return answer == "y"


def print_flags(flags, write_file=True):
    """
    Prints the flag arguments of the current execution

    Parameters
    ----------
    flags : argparse.Namespace
        Collection of flags (arguments)
    """
    print('')
    print('--------------------------------------------------')
    print('Running %s' % sys.argv[0])
    print('')
    print('FLAGS:')
    for f in vars(flags):
        print('{}: {}'.format(f, getattr(flags, f)))
    print('')


def write_flags(flags, write_file=True):
    """
    Writes the flag arguments of the current execution into a txt file.

    Parameters
    ----------
    flags : argparse.Namespace
        Collection of flags (arguments)
    """
    output_file = os.path.join(flags.train_dir, 'flags_' +
                               time.strftime('%a_%d_%b_%Y_%H%M%S'))
    with open(output_file, 'w') as file:
        file.write('FLAGS:\n')
        for f in vars(flags):
            file.write('{}: {}\n'.format(f, getattr(flags, f)))


def print_test_results(d):

    if 'test' in d:
        sys.stdout.write('\nTest performance:\n')
        test_dict = d['test']

        if 'orig' in test_dict:
            sys.stdout.write('\n\tOriginal images:\n')
            for metric, metric_dict in test_dict['orig']['metrics'].items():
                sys.stdout.write('\t\t{}: {:.4f}\n'.format(
                    metric, metric_dict['mean']))

        if 'daug' in test_dict:
            sys.stdout.write('\n\tData augmentation:\n')
            for scheme, scheme_dict in test_dict['daug'].items():
                sys.stdout.write('\t\t{}:\n'.format(scheme))
                for metric, metric_dict in scheme_dict['metrics'].items():
                    sys.stdout.write('\t\t\t{} - mean: {:.4f}\n'.format(
                        metric, metric_dict['mean']))
                    sys.stdout.write('\t\t\t{} - median: {:.4f}\n'.format(
                        metric, metric_dict['median']))
                    if len(metric_dict['per_rep']) > 1:
                        sys.stdout.write('\t\t\t{} per rep: '.format(metric))
                        for acc in metric_dict['per_rep']:
                            sys.stdout.write('{:.4f} '.format(acc))
                        sys.stdout.write('\n')
                if len(metric_dict['per_rep']) > 1:
                    sys.stdout.write('\t\t\tmean std predictions: '
                                     '{:.4f}\n'.format(
                                         scheme_dict['mean_std_predictions']))

    if 'train' in d:
        sys.stdout.write('\nTrain performance:\n')
        train_dict = d['train']

        if 'orig' in train_dict:
            sys.stdout.write('\n\tOriginal images:\n')
            for metric, metric_dict in train_dict['orig']['metrics'].items():
                sys.stdout.write('\t\t{}: {:.4f}\n'.format(
                    metric, metric_dict['mean']))

        if 'daug' in train_dict:
            sys.stdout.write('\n\tData augmentation:\n')
            for scheme, scheme_dict in train_dict['daug'].items():
                sys.stdout.write('\t\t{}:\n'.format(scheme))
                for metric, metric_dict in scheme_dict['metrics'].items():
                    sys.stdout.write('\t\t\t{} - mean: {:.4f}\n'.format(
                        metric, metric_dict['mean']))
                    sys.stdout.write('\t\t\t{} - median: {:.4f}\n'.format(
                        metric, metric_dict['median']))
                    if len(metric_dict['per_rep']) > 1:
                        sys.stdout.write('\t\t\t{} per rep: '.format(metric))
                        for acc in metric_dict['per_rep']:
                            sys.stdout.write('{:.4f} '.format(acc))
                        sys.stdout.write('\n')
                if len(metric_dict['per_rep']) > 1:
                    sys.stdout.write('\t\t\tmean std predictions: '
                                     '{:.4f}\n'.format(
                                         scheme_dict['mean_std_predictions']))

    if 'ablation' in d:
        if 'test' in d['ablation']:
            sys.stdout.write('\nRobustness to ablation of units '
                             '(test data):\n')
            for pct in sorted(d['ablation']['test'].keys()):
                pct_dict = d['ablation']['test'][pct]
                sys.stdout.write('\t{} % of the units:\n'.format(100 * pct))
                for metric, metric_dict in pct_dict['metrics'].items():
                    sys.stdout.write('\t\t{} - mean: {:.4f}\n'.format(
                        metric, metric_dict['mean']))
                    sys.stdout.write('\t\t{} - std: {:.4f}\n'.format(
                        metric, metric_dict['std']))
                    if len(metric_dict['per_rep']) > 1:
                        sys.stdout.write('\t\t{} per rep: '.format(metric))
                        for acc in metric_dict['per_rep']:
                            sys.stdout.write('{:.4f} '.format(acc))
                        sys.stdout.write('\n')

        if 'train' in d['ablation']:
            sys.stdout.write('\nRobustness to ablation of units '
                             '(train data):\n')
            for pct in sorted(d['ablation']['train'].keys()):
                pct_dict = d['ablation']['train'][pct]
                sys.stdout.write('\t{} % of the units:\n'.format(100 * pct))
                for metric, metric_dict in pct_dict['metrics'].items():
                    sys.stdout.write('\t\t{} - mean: {:.4f}\n'.format(
                        metric, metric_dict['mean']))
                    sys.stdout.write('\t\t{} - std: {:.4f}\n'.format(
                        metric, metric_dict['std']))
                    if len(metric_dict['per_rep']) > 1:
                        sys.stdout.write('\t\t{} per rep: '.format(metric))
                        for acc in metric_dict['per_rep']:
                            sys.stdout.write('{:.4f} '.format(acc))
                        sys.stdout.write('\n')

    if 'adv' in d:
        sys.stdout.write('\nRobustness to adversarial examples '
                         '- white box attacks:\n')
        for attack, attack_dict in d['adv']['white_box'].items():
            sys.stdout.write('\tAttack: {}:\n'.format(attack))
            if 'mean_acc' in attack_dict:
                sys.stdout.write('\t\tmean acc: {:.4f}\n'.format(
                    attack_dict['mean_acc']))
                sys.stdout.write('\t\tmean mse: {:.4f}\n'.format(
                    attack_dict['mean_mse']))
            else:
                for eps, eps_dict in sorted(
                        d['adv']['white_box'][attack].items()):
                    sys.stdout.write('\t\tepsilon = {}:\n'.format(eps))
                    sys.stdout.write('\t\t\tmean acc: {:.4f}\n'.format(
                        eps_dict['mean_acc']))
                    sys.stdout.write('\t\t\tmean mse: {:.4f}\n'.format(
                        eps_dict['mean_mse']))
        if 'black_box' in d['adv']:
            sys.stdout.write('Robustness to adversarial examples - '
                  'black box attacks:\n')
            for attack, attack_dict in d['adv']['black_box'].items():
                sys.stdout.write('\tAttack: {}:\n'.format(attack))
                if 'mean_acc' in attack_dict:
                    sys.stdout.write('\t\tmean acc: {:.4f}\n'.format(
                        attack_dict['mean_acc']))
                    sys.stdout.write('\t\tmean mse: {:.4f}\n'.format(
                        attack_dict['mean_mse']))
                else:
                    for eps, eps_dict in sorted(
                            d['adv']['black_box'][attack].items()):
                        sys.stdout.write('\t\tepsilon = {}:\n'.format(eps))
                        sys.stdout.write('\t\t\tmean acc: {:.4f}\n'.format(
                            eps_dict['mean_acc']))
                        sys.stdout.write('\t\t\tmean mse: {:.4f}\n'.format(
                            eps_dict['mean_mse']))

    if 'activations' in d:
        sys.stdout.write('\nNorm of activations [mean (std)]:\n')
        for layer, layer_dict in sorted(d['activations']['summary'].items()):
            sys.stdout.write('\t{}:\n'.format(layer))
            for norm_key, norm_dict in layer_dict.items():
                sys.stdout.write('\t\t{}: {:.4f} ({:.4f})\n'.format(
                    norm_key, norm_dict['mean'], norm_dict['std']))

    sys.stdout.flush()


def write_test_results(d, output_file, write_mode='w'):
    f =  open(output_file, write_mode)

    if 'test' in d:
        f.write('\nTest performance:\n')
        test_dict = d['test']

        if 'orig' in test_dict:
            f.write('\n\tOriginal images:\n')
            for metric, metric_dict in test_dict['orig']['metrics'].items():
                f.write('\t\t{}: {:.4f}\n'.format(
                    metric, metric_dict['mean']))

        if 'daug' in test_dict:
            f.write('\n\tData augmentation:\n')
            for scheme, scheme_dict in test_dict['daug'].items():
                f.write('\t\t{}:\n'.format(scheme))
                for metric, metric_dict in scheme_dict['metrics'].items():
                    f.write('\t\t\t{} - mean: {:.4f}\n'.format(
                        metric, metric_dict['mean']))
                    f.write('\t\t\t{} - median: {:.4f}\n'.format(
                        metric, metric_dict['median']))
                    if len(metric_dict['per_rep']) > 1:
                        f.write('\t\t\t{} per rep: '.format(metric))
                        for acc in metric_dict['per_rep']:
                            f.write('{:.4f} '.format(acc))
                        f.write('\n')
                if len(metric_dict['per_rep']) > 1:
                    f.write('\t\t\tmean std predictions: {:.4f}\n'.format(
                        scheme_dict['mean_std_predictions']))

    if 'train' in d:
        f.write('\nTrain performance:\n')
        train_dict = d['train']

        if 'orig' in train_dict:
            f.write('\n\tOriginal images:\n')
            for metric, metric_dict in train_dict['orig']['metrics'].items():
                f.write('\t\t{}: {:.4f}\n'.format(
                    metric, metric_dict['mean']))

        if 'daug' in train_dict:
            f.write('\n\tData augmentation:\n')
            for scheme, scheme_dict in train_dict['daug'].items():
                f.write('\t\t{}:\n'.format(scheme))
                for metric, metric_dict in scheme_dict['metrics'].items():
                    f.write('\t\t\t{} - mean: {:.4f}\n'.format(
                        metric, metric_dict['mean']))
                    f.write('\t\t\t{} - median: {:.4f}\n'.format(
                        metric, metric_dict['median']))
                    if len(metric_dict['per_rep']) > 1:
                        f.write('\t\t\t{} per rep: '.format(metric))
                        for acc in metric_dict['per_rep']:
                            f.write('{:.4f} '.format(acc))
                        f.write('\n')
                if len(metric_dict['per_rep']) > 1:
                    f.write('\t\t\tmean std predictions: {:.4f}\n'.format(
                        scheme_dict['mean_std_predictions']))

    if 'ablation' in d:
        if 'test' in d['ablation']:
            f.write('\nRobustness to ablation of units (test data):\n')
            for pct in sorted(d['ablation']['test'].keys()):
                pct_dict = d['ablation']['test'][pct]
                f.write('\t{} % of the units:\n'.format(100 * pct))
                for metric, metric_dict in pct_dict['metrics'].items():
                    f.write('\t\t{} - mean: {:.4f}\n'.format(
                        metric, metric_dict['mean']))
                    f.write('\t\t{} - std: {:.4f}\n'.format(
                        metric, metric_dict['std']))
                    if len(metric_dict['per_rep']) > 1:
                        f.write('\t\t{} per rep: '.format(metric))
                        for acc in metric_dict['per_rep']:
                            f.write('{:.4f} '.format(acc))
                        f.write('\n')

        if 'train' in d['ablation']:
            f.write('\nRobustness to ablation of units (train data):\n')
            for pct in sorted(d['ablation']['train'].keys()):
                pct_dict = d['ablation']['train'][pct]
                f.write('\t{} % of the units:\n'.format(100 * pct))
                for metric, metric_dict in pct_dict['metrics'].items():
                    f.write('\t\t{} - mean: {:.4f}\n'.format(
                        metric, metric_dict['mean']))
                    f.write('\t\t{} - std: {:.4f}\n'.format(
                        metric, metric_dict['std']))
                    if len(metric_dict['per_rep']) > 1:
                        f.write('\t\t{} per rep: '.format(metric))
                        for acc in metric_dict['per_rep']:
                            f.write('{:.4f} '.format(acc))
                        f.write('\n')

    if 'adv' in d:
        f.write('\nRobustness to adversarial examples - white box attacks:\n')
        for attack, attack_dict in d['adv']['white_box'].items():
            f.write('\tAttack: {}:\n'.format(attack))
            if 'mean_acc' in attack_dict:
                f.write('\t\tmean acc: {:.4f}\n'.format(
                    attack_dict['mean_acc']))
                f.write('\t\tmean mse: {:.4f}\n'.format(
                    attack_dict['mean_mse']))
            else:
                for eps, eps_dict in sorted(
                        d['adv']['white_box'][attack].items()):
                    f.write('\t\tepsilon = {}:\n'.format(eps))
                    f.write('\t\t\tmean acc: {:.4f}\n'.format(
                        eps_dict['mean_acc']))
                    f.write('\t\t\tmean mse: {:.4f}\n'.format(
                        eps_dict['mean_mse']))
        if 'black_box' in d['adv']:
            f.write('Robustness to adversarial examples - '
                  'black box attacks:\n')
            for attack, attack_dict in d['adv']['black_box'].items():
                f.write('\tAttack: {}:\n'.format(attack))
                if 'mean_acc' in attack_dict:
                    f.write('\t\tmean acc: {:.4f}\n'.format(
                        attack_dict['mean_acc']))
                    f.write('\t\tmean mse: {:.4f}\n'.format(
                        attack_dict['mean_mse']))
                else:
                    for eps, eps_dict in sorted(
                            d['adv']['black_box'][attack].items()):
                        f.write('\t\tepsilon = {}:\n'.format(eps))
                        f.write('\t\t\tmean acc: {:.4f}\n'.format(
                            eps_dict['mean_acc']))
                        f.write('\t\t\tmean mse: {:.4f}\n'.format(
                            eps_dict['mean_mse']))

    if 'activations' in d:
        f.write('\nNorm of activations [mean (std)]:\n')
        for layer, layer_dict in sorted(d['activations']['summary'].items()):
            f.write('\t{}:\n'.format(layer))
            for norm_key, norm_dict in layer_dict.items():
                f.write('\t\t{}: {:.4f} ({:.4f})\n'.format(
                    norm_key, norm_dict['mean'], norm_dict['std']))

    f.close()


def define_train_params(train_config, output_dir=None):

    train = train_config.train
    daug = train_config.daug
    data = train_config.data

    initial_lr_orig = train.lr.init_lr
    batch_size_orig = train.batch_size.tr

    # Train parameters
    if daug.aug_per_img_tr > 1:
        if train.simulate.norep_samples:
            if train.simulate.bs_lr.lower() == 'bs':
                train.batch_size.tr = batch_size_orig * daug.aug_per_img_tr
                train.lr.init_lr = initial_lr_orig
            elif train.simulate.bs_lr.lower() == 'lr':
                train.lr.init_lr = initial_lr_orig / daug.aug_per_img_tr
                train.batch_size.tr = batch_size_orig
            else:
                raise ValueError('simulate_bs_lr can only be bs or lr')
        else:
            train.batch_size.tr = batch_size_orig
            train.lr.init_lr = initial_lr_orig

        # Define the batch size for the generator
        n_diff_batch_tr = train.batch_size.tr / float(daug.aug_per_img_tr)
        if n_diff_batch_tr.is_integer():
            train.batch_size.gen_tr = int(n_diff_batch_tr)
        else:
            raise ValueError('aug_per_im must be a divisor of batch_size')
    else:
        if train.simulate.rep_samples:
            if train.simulate.bs_lr.lower() == 'bs':
                new_batch_size = \
                        batch_size_orig / float(train.simulate.rep_samples)
                if new_batch_size.is_integer():
                    train.batch_size.tr = int(new_batch_size)
                else:
                    raise ValueError('simulate_rep_samples must be a divisor '
                                     'of batch_size')
                train.lr.init_lr = initial_lr_orig
            elif train.simulate.bs_lr.lower() == 'lr':
                train.lr.init_lr = initial_lr_orig * train.simulate.rep_samples
                train.batch_size.tr = batch_size_orig
            else:
                raise ValueError('simulate_bs_lr can only be bs or lr')
        else:
            train.batch_size.tr = batch_size_orig
            train.lr.init_lr = initial_lr_orig
        n_diff_batch_tr = train.batch_size.tr
        train.batch_size.gen_tr = n_diff_batch_tr

    # Set number of epochs and number of iterations per epoch
    train.epochs_orig = train.epochs
    if train.simulate.true_epochs & (train.batch_size.tr != n_diff_batch_tr):
        train.batches_per_epoch_tr = \
                int(np.ceil(float(data.n_train) / n_diff_batch_tr))
        train.epochs = int(np.ceil(
            float(train.epochs * n_diff_batch_tr) / batch_size_orig))
    else:
        train.batches_per_epoch_tr = \
                int(np.ceil(float(data.n_train) / train.batch_size.tr))

    # Validation parameters
    n_diff_batch_val = batch_size_orig / float(daug.aug_per_img_val)
    if n_diff_batch_val.is_integer():
        train.batch_size.gen_val = int(n_diff_batch_val)
    else:
        raise ValueError('aug_per_img_val must be a divisor of batch_size')
    
    # Compute number of batches per validation epoch
    # Computed with respect to the true batch size, in order to avoid too many
    # iterations
    train.batches_per_epoch_val = \
            int(np.ceil(float(data.n_val) / train.batch_size.val))

    # Determine whether invariance training is performed
    if (daug.aug_per_img_tr > 1) | (daug.aug_per_img_val > 1):
        train_config.optimizer.invariance = True
    
    # Print training parameters
    if output_dir:
        write_file = True
    print_training_params(initial_lr_orig, train.lr.init_lr, batch_size_orig, 
                          train.batch_size.tr, daug.aug_per_img_tr,
                          n_diff_batch_tr, train.epochs,
                          train.batches_per_epoch_tr, output_dir, write_file)

    # Set the sub-namespaces within the train config namespace
    train_config.train = train
    train_config.daug = daug
    train_config.data = data

    return train_config


def print_training_params(learning_rate_orig, learning_rate_new, 
                          batch_size_orig, batch_size_tr, n_rep_samples,
                          n_diff_samples, n_epochs, steps_per_epoch,
                          train_dir, write_file=True):
    print('--------------------------------------------------')
    print('Original learning rate: {}'.format(learning_rate_orig))
    print('Original batch size: {}'.format(batch_size_orig))
    if n_rep_samples > 1:
        print('Repeated samples within the batches: Yes')
        if batch_size_orig != batch_size_tr:
            print('Simulate no repeated samples: Yes (batch size)')
            print('\tActual training batch size: {}'.format(batch_size_tr))
        elif learning_rate_new != learning_rate_orig:
            print('Simulate no repeated samples: Yes (learning rate)')
            print('\tNew learning rate: {}'.format(learning_rate_new))
        else:
            print('Simulate no repeated samples: No')
        print('\tNumber of repeated samples: {}'.format(n_rep_samples))
        print('\tNumber of different samples: {}'.format(int(n_diff_samples)))
    else:
        print('Repeated samples within the batches: No')
        if batch_size_orig != batch_size_tr:
            print('Simulate repeated samples: Yes (batch size)')
            print('\tActual training batch size: {}'.format(batch_size_tr))
        elif learning_rate_new != learning_rate_orig:
            print('Simulate repeated samples: Yes (learning rate)')
            print('\tNew learning rate: {}'.format(learning_rate_new))
        else:
            print('Simulate repeated samples: No')
    print('Number of epochs: {}'.format(n_epochs))
    print('Number of iterations per epoch: {}'.format(steps_per_epoch))
    n_iterations = steps_per_epoch * n_epochs
    n_samples_seen = n_iterations * batch_size_tr
    print('Total number of iterations: {}'.format(n_iterations))
    print('Total number of samples seen: {}'.format(n_samples_seen))
    print('')

    if write_file:
        output_file = os.path.join(train_dir, 'train_params_' +
                                   time.strftime('%a_%d_%b_%Y_%H%M%S'))
        with open(output_file, 'w') as f:
            f.write('TRAINING PARAMETERS:\n')
            f.write('Original learning rate: {}\n'.format(learning_rate_orig))
            f.write('Original batch size: {}\n'.format(batch_size_orig))
            if n_rep_samples > 1:
                f.write('Repeated samples within the batches: Yes\n')
                if batch_size_orig != batch_size_tr:
                    f.write('Simulate no repeated samples: Yes\n')
                    f.write('\tNew learning rate: {}\n'.format(learning_rate_new))
                    f.write('\tActual training batch size: {}\n'.format(batch_size_tr))
                else:
                    f.write('Simulate no repeated samples: No\n')
                    if learning_rate_new != learning_rate_orig:
                        f.write('\tNew learning rate: {}\n:'.format(learning_rate_new))
                f.write('\tNumber of repeated samples: {}\n'.format(n_rep_samples))
                f.write('\tNumber of different samples: {}\n'.format(int(n_diff_samples)))
            else:
                f.write('Repeated samples within the batches: No\n')
                if batch_size_orig != batch_size_tr:
                    f.write('Simulate repeated samples: Yes\n')
                    f.write('\tNew learning rate: {}\n'.format(learning_rate_new))
                    f.write('\tActual training batch size: {}\n'.format(batch_size_tr))
                else:
                    f.write('Simulate repeated samples: No\n')
                    if learning_rate_new != learning_rate_orig:
                        f.write('\tNew learning rate: {}\n:'.format(learning_rate_new))
            f.write('Number of epochs: {}\n'.format(n_epochs))
            f.write('Number of iterations per epoch: {}\n'.format(steps_per_epoch))
            n_iterations = steps_per_epoch * n_epochs
            f.write('Total number of iterations: {}\n'.format(n_iterations))


def write_tensorboard(callback, names, values, step):
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, step)
        callback.writer.flush()


class LossWeightsScheduler(Callback):
    def __init__(self,
                 loss_weights,
                 decay_epochs_daug,
                 decay_epochs_class,
                 decay_rate_daug,
                 decay_rate_class,
                 epochs_pretraining_daug=0,
                 epochs_pretraining_class=0):
        super(LossWeightsScheduler, self).__init__()
        self.loss_weights = loss_weights
        self.decay_epochs_daug = decay_epochs_daug
        self.decay_epochs_class = decay_epochs_class
        if isinstance(decay_rate_daug, list):
            self.decay_rate_daug = decay_rate_daug
        else:
            self.decay_rate_daug = \
                    [decay_rate_daug] * len(self.decay_epochs_daug)
        if isinstance(decay_rate_class, list):
            self.decay_rate_class = decay_rate_class
        else:
            self.decay_rate_class = \
                    [decay_rate_class] * len(self.decay_epochs_class)

        # Configure invariance pre-training. That is, the weights of all losses
        # except the selected for pre-training are set to 0 for the given
        # number of epochs, where they are set to the original initial loss.
        # Then, the invariance losses will be decayed by the specified rates
#         if epochs_pretraining_daug: loss_daug = _get_total_loss('daug_inv')
#         else:
#             loss_daug = 0.
#         if epochs_pretraining_class:
#             loss_class = _get_total_loss('class_inv')
#         else:
#             loss_class = 0.
#         init_loss_daug = 1. - loss_class / 2.
#         init_loss_class = 1. - loss_daug / 2.

    def on_epoch_begin(self, epoch, logs={}):

        def _get_total_loss(keyword):
            total_loss = K.variable(0.)
            for loss_name, loss_weight in self.loss_weights.items():
                if keyword in loss_name:
                    total_loss = total_loss + loss_weight
            return total_loss
                
        if epoch in self.decay_epochs_daug:
            decay = self.decay_rate_daug[self.decay_epochs_daug.index(epoch)]
            for loss_name, loss_weight in self.loss_weights.items():
                if 'daug_inv' in loss_name:
                    K.set_value(self.loss_weights[loss_name], 
                                K.get_value(loss_weight) * decay)
        if epoch in self.decay_epochs_class:
            decay = self.decay_rate_class[self.decay_epochs_class.index(epoch)]
            for loss_name, loss_weight in self.loss_weights.items():
                if 'class_inv' in loss_name:
                    K.set_value(self.loss_weights[loss_name], 
                                K.get_value(loss_weight) * decay)

        # Update softmax loss
        loss_daug = _get_total_loss('daug_inv')
        loss_class = _get_total_loss('class_inv')
        K.set_value(self.loss_weights['softmax'], 
                    1. - K.get_value(loss_daug + loss_class))

    def on_epoch_end(self, epoch, logs={}):
        pass


class Gradients(Callback):
    def __init__(self):
        super(Gradients, self).__init__()
#         self.layer_name_pattern = 'conv[0-9][0-9]?(relu|bn)?$'
        self.layer_name_pattern = 'conv[0-9][0-9]??$'
                
    def on_train_begin(self, logs={}):
        self.w_dict = {}
        self.delta_w_dict = {}
        for layer in self.model.layers:
            if re.match(self.layer_name_pattern, layer.name):
                self.w_dict.update({layer.name: {}})
                self.delta_w_dict.update({layer.name: {}})
                for weight in layer.weights:
                    w_name = weight.name.replace(':', '_')
                    self.w_dict[layer.name].update({w_name: weight})
                    self.delta_w_dict[layer.name].update({w_name: []})

    def on_batch_end(self, batch, logs={}):
        for layer in self.model.layers:
            if re.match(self.layer_name_pattern, layer.name):
                for weight in layer.weights:
                    w_name = weight.name.replace(':', '_')
                    norm = K.sum(K.square(K.flatten(weight) - K.flatten(self.w_dict[layer.name][w_name])))
                    self.w_dict[layer.name][w_name] = weight
                    self.delta_w_dict[layer.name][w_name].append(K.get_value(norm))


class RDM(Callback):
    def __init__(self, 
                 train_dir=None,
                 hdf5_file='~/datasets/hdf5/fmri_images_128x128.hdf5',
                 group='fmri',
                 dataset='data',
#                  layers_regex='conv[0-9][0-9]??$'):
                 layers_regex='.*relu$'):
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.dataset = np.asarray(self.hdf5_file[group][dataset])
        self.layers_regex = layers_regex
        
        self.log_dir = os.path.join(train_dir, 'rdms')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            raise ValueError('RDMs output directory already exists!')

        super(RDM, self).__init__()
                
    def on_epoch_end(self, epoch, logs=None):
        rdms_dict = {}
        for layer in self.model.layers:
            if re.match(self.layers_regex, layer.name):
                activation_function = K.function([self.model.input, 
                                                  K.learning_phase()], 
                                                 [layer.output])
                activation = activation_function([self.dataset, 0])[0]
                activation = np.reshape(activation, [activation.shape[0], -1])
                rdms_dict.update({layer.name: 1.0 - np.corrcoef(activation)})
                pickle.dump(rdms_dict, open(os.path.join(
                    self.log_dir, 'rdms_e{:03d}.p'.format(epoch)), 'wb'))


class SumOfMseLosses(Callback):
    def __init__(self):
        super(SumOfMseLosses, self).__init__()
                

    def on_epoch_end(self, epoch, logs=None):
        mse_metrics = [metric for metric in self.model.metrics_tensors 
                       if 'mse' in metric.name]
        sum_mse = K.sum(mse_metrics)
        self.model.metrics_tensors.append(sum_mse)


class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n\tlr: {}'.format(K.eval(self.model.optimizer.lr)))

    def on_batch_end(self, batch, logs=None):
        print('\n\tlr: {}'.format(K.eval(self.model.optimizer.lr)))

