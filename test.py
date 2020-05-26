"""
Routines for testing a model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import dask.array as da
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras.models import Model
import h5py
import yaml
import sys
import os
import argparse
from tqdm import tqdm
import re
from time import time

from data_input import train_val_split, subsample_data
from data_input import get_generator, batch_generator
from data_input import generate_batches
from adv_utils import init_attack
from surgery import ablate_activations, del_mse_nodes, del_extra_nodes
from surgery import network2dict, restore_nodes

from utils import print_flags
from utils import pairwise_loss, invariance_loss, mean_loss
from utils import handle_metrics
from utils import prepare_test_config, numpy_to_python
from utils import print_test_results, write_test_results
## Import the whole compat version of keras to set the losses =================
import tensorflow.compat.v1.keras as keras
## ============================================================================
keras.losses.pairwise_loss = pairwise_loss
keras.losses.invariance_loss = invariance_loss
keras.losses.mean_loss = mean_loss

# Initialize the Flags container
FLAGS = None


def main(argv=None):

    K.set_floatx('float32')

    print_flags(FLAGS)

    # Read or/and prepare test config dictionary
    if FLAGS.test_config_file:
        with open(FLAGS.test_config_file, 'r') as yml_file:
            test_config = yaml.load(yml_file, Loader=yaml.FullLoader)
    else:
        test_config = {}
    test_config = prepare_test_config(test_config, FLAGS)

    # Load model
    model = load_model(os.path.join(FLAGS.model))

    # Open HDF5 file containing the data set and get images and labels
    hdf5_file = h5py.File(FLAGS.data_file, 'r')
    images_tr, images_tt, labels_tr, labels_tt, _ = train_val_split(
            hdf5_file, FLAGS.group_tr, FLAGS.group_tt, FLAGS.chunk_size)

    # Test
    results_dict = test(images_tt, labels_tt, images_tr, labels_tr, model,
                        test_config, FLAGS.batch_size, FLAGS.chunk_size)

    # Print and write results
    if FLAGS.output_dir:

        if FLAGS.output_dir == '-1':
            FLAGS.output_dir = os.path.dirname(FLAGS.model)

        if FLAGS.append:
            write_mode = 'a'
        else:
            write_mode = 'w'

        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        output_file = os.path.join(FLAGS.output_dir,
                                   '{}.txt'.format(FLAGS.output_basename))
        write_test_results(results_dict, output_file, write_mode)
        output_file = os.path.join(FLAGS.output_dir, 
                                   '{}.yml'.format(FLAGS.output_basename))
        with open(output_file, write_mode) as f:
            results_dict = numpy_to_python(results_dict)
            yaml.dump(results_dict, f, default_flow_style=False)
    print_test_results(results_dict)

    # Close HDF5 File
    hdf5_file.close()


def test(images_tt, labels_tt, images_tr, labels_tr, model, test_config,
         batch_size, chunk_size):
    """
    Performs a set of test operations, as specified in test_config.

    Parameters
    ----------
    images_tt : h5py Dataset
        The set of test images

    labels_tt : h5py Dataset
        The ground truth labels of the test set

    images_tr : h5py Dataset
        The set of train images

    labels_tr : h5py Dataset
        The ground truth labels of the train set

    model : Keras Model
        The model

    batch_size : int
        Batch size

    test_config : str
        YAML file specifying the aspects to test and their parameters

    Returns
    -------
    results_dict : dict
        Dictionary containing some performance metrics
    """
    # Ensure the model has no MSE nodes and outputs
    model = del_mse_nodes(model)

    results_dict = {}
    daug_params_dicts = {}

    # Test performance
    if 'test' in test_config:
        results_dict.update({'test': {}})
        test_config_test = test_config['test']

        # Original images (no data augmentation)
        if 'orig' in test_config_test:
            print('\nComputing test performance with the original images')
            results_dict['test'].update({'orig': {}})
            results_dict['test']['orig'] = test_rep(
                    images_tt, labels_tt, batch_size, model,
                    test_config_test['orig']['daug_params'], 1,
                    test_config_test['orig']['metrics'])

        # Augmented images
        if 'daug' in test_config_test:
            results_dict['test'].update({'daug': {}})
            for scheme in test_config_test['daug']:
                print('\nComputing test performance with {} '
                      'augmentation'.format(scheme))
                results_dict['test']['daug'].update({scheme: {}})
                results_dict['test']['daug'][scheme] = test_rep(
                        images_tt, labels_tt, batch_size, model,
                        test_config_test['daug'][scheme]['daug_params'],
                        test_config_test['daug'][scheme]['repetitions'],
                        test_config_test['daug'][scheme]['metrics'])

    # Train performance
    if 'train' in test_config:
        results_dict.update({'train': {}})
        test_config_train = test_config['train']

        # Original images (no data augmentation)
        if 'orig' in test_config_train:
            print('\nComputing train performance with the original images')
            results_dict['train'].update({'orig': {}})
            results_dict['train']['orig'] = test_rep(
                    images_tr, labels_tr, batch_size, model,
                    test_config_train['orig']['daug_params'], 1,
                    test_config_train['orig']['metrics'])

        # Augmented images
        if 'daug' in test_config_train:
            results_dict['train'].update({'daug': {}})
            for scheme in test_config_train['daug']:
                print('\nComputing train performance with {} '
                      'augmentation'.format(scheme))
                results_dict['train']['daug'].update({scheme: {}})
                results_dict['train']['daug'][scheme] = test_rep(
                        images_tr, labels_tr, batch_size, model,
                        test_config_train['daug'][scheme]['daug_params'],
                        test_config_train['daug'][scheme]['repetitions'],
                        test_config_train['daug'][scheme]['metrics'])

    # Test robustness to ablation of units
    if 'ablation' in test_config:
        results_dict.update({'ablation': {}})
        # Test set
        if 'test' in test_config['ablation']:
            results_dict['ablation'].update({'test': {}})
            for pct in test_config['ablation']['pct']:
                print('\nComputing test robustness to ablation of {} % of the '
                      'units'.format(100 * pct))
                results_dict['ablation']['test'].update({pct: {}})
                results_dict['ablation']['test'][pct] = test_ablation(
                        images_tt, labels_tt, batch_size, model,
                        test_config['ablation']['daug_params'], 
                        test_config['ablation']['repetitions'],
                        test_config['ablation']['layer_regex'],
                        pct,
                        test_config['ablation']['seed'],
                        test_config['ablation']['metrics'])

        # Train set
        if 'train' in test_config['ablation']:
            results_dict['ablation'].update({'train': {}})
            for pct in test_config['ablation']['pct']:
                print('\nComputing train robustness to ablation of {} % of '
                      'the units'.format(100 * pct))
                results_dict['ablation']['train'].update({pct: {}})
                results_dict['ablation']['train'][pct] = test_ablation(
                        images_tr, labels_tr, batch_size, model,
                        test_config['ablation']['daug_params'], 
                        test_config['ablation']['repetitions'],
                        test_config['ablation']['layer_regex'],
                        pct,
                        test_config['ablation']['seed'],
                        test_config['ablation']['metrics'])

    # Test adversarial robustness
    if 'adv' in test_config:
        results_dict.update({'adv': {}})

        # Subsample data
        images_adv, labels_adv, aux_hdf5 = subsample_data(
                images_tt, labels_tt, test_config['adv']['pct_data'], 
                chunk_size, test_config['adv']['shuffle_data'],
                test_config['adv']['shuffle_seed'])

        # White box attack
        results_dict['adv'].update({'white_box': {}})
        adv_model = model
        for attack, attack_dict in test_config['adv']['attacks'].items():
            print('\nComputing white box adversarial robustness '
                  'towards {}'.format(attack))
            results_dict['adv']['white_box'].update({attack: {}})
            results_dict_attack = results_dict['adv']['white_box'][attack]
            if 'eps' in attack_dict and \
               isinstance(attack_dict['eps'], list):
                epsilons = attack_dict['eps']
                if 'eps_iter' in attack_dict:
                    epsilons_iter = attack_dict['eps_iter']
                else:
                    epsilons_iter = [None] * len(epsilons)
                for eps, eps_iter in zip(epsilons, epsilons_iter):
                    results_dict_attack.update({eps: {}})
                    attack_dict['eps'] = eps
                    if eps_iter:
                        attack_dict['eps_iter'] = eps_iter
                    results_dict_attack[eps] = test_adv(
                            images_adv, labels_adv, batch_size, model,
                            adv_model, test_config['adv']['daug_params'],
                            attack_dict)
                attack_dict['eps'] = epsilons
                if 'eps_iter' in attack_dict:
                    attack_dict['eps_iter'] = epsilons_iter
            else:
                results_dict_attack = test_adv(
                        images_adv, labels_adv, batch_size, model, adv_model,
                        test_config['adv']['daug_params'], 
                        attack_dict)

        # Black box attack
        if test_config['adv']['black_box_model']:
            adv_model = load_model(test_config['adv']['black_box_model'])
            results_dict['adv'].update({'black_box': {}})
            for attack, attack_dict in test_config['adv']['attacks'].items():
                print('\nComputing black box adversarial robustness '
                      'towards {}'.format(attack))
                results_dict['adv']['black_box'].update({attack: {}})
                results_dict_attack = results_dict['adv']['black_box'][attack]
                if 'eps' in attack_dict and \
                   isinstance(attack_dict['eps'], list):
                    epsilons = attack_dict['eps']
                    if 'eps_iter' in attack_dict:
                        epsilons_iter = attack_dict['eps_iter']
                    else:
                        epsilons_iter = [None] * len(epsilons)
                    for eps, eps_iter in zip(epsilons, epsilons_iter):
                        results_dict_attack.update({eps: {}})
                        attack_dict['eps'] = eps
                        if eps_iter:
                            attack_dict['eps_iter'] = eps_iter
                        results_dict_attack[eps] = test_adv(
                                images_adv, labels_adv, batch_size, model,
                                adv_model, test_config['adv']['daug_params'],
                                attack_dict)
                    attack_dict['eps'] = epsilons
                    if 'eps_iter' in attack_dict:
                        attack_dict['eps_iter'] = epsilons_iter
                else:
                    results_dict_attack = test_adv(
                            images_adv, labels_adv, batch_size, model,
                            adv_model, test_config['adv']['daug_params'],
                            attack_dict)
    else:
        aux_hdf5 = []

    # Compute norms and metrics from the activations
    if 'activations' in test_config:
        print('\nComputing metrics related to the activations')
        results_dict.update({'activations': {}})
        results_dict['activations'] = activations(
                images_tt, labels_tt, batch_size, model, 
                test_config['activations']['layer_regex'],
                test_config['activations']['nodaug_params'],
                test_config['activations']['daug_params'],
                test_config['activations']['include_input'],
                test_config['activations']['class_invariance'],
                test_config['activations']['n_daug_rep'],
                test_config['activations']['norms'])

    for f in aux_hdf5:
        filename = f.filename
        f.close()
        os.remove(filename)

    return results_dict


def test_rep(images, labels, batch_size, model, daug_params, repetitions,
             metrics=['accuracy']):
    """
    Tests the performance of a model on a set of images, transformed according
    to the specified augmentation parameters, and computes statistics over a
    number of repetitions.

    Parameters
    ----------
    images : h5py Dataset
        The set of images

    labels : h5py Dataset
        The ground truth labels

    batch_size : int
        Batch size

    model : Keras Model
        The model

    daug_params : dict
        Dictionary of data augmentation parameters

    repetitions : int
        Number of data augmentation repetitions

    Returns
    -------
    results_dict : dict
        Dictionary containing some performance metrics
    """
    n_images = images.shape[0]
    n_classes = labels.shape[1]
    n_batches_per_epoch = int(np.ceil(float(n_images) / batch_size))

    # Create batch generator
    image_gen = get_generator(images, **daug_params)
    batch_gen = batch_generator(image_gen, images, labels, batch_size, 
                                aug_per_im=1, shuffle=False)

    # Initialize matrix to store the predictions.
    predictions = np.zeros([n_images, n_classes, repetitions])

    # Iterate over the random repetitions
    for r in range(repetitions):
        print('Run %d/%d' % (r+1, repetitions))
        init = 0
        batch_gen.image_gen.reset()
        # Iterate over the whole data set batch by batch
        for _ in tqdm(range(n_batches_per_epoch)):
            batch_images, _ = next(batch_gen())
            batch_size = batch_images.shape[0]
            end = init + batch_size
            predictions[init:end, :, r] = \
                    model.predict_on_batch(batch_images)
            init = end

    results_dict = _stats_from_pred(predictions, labels, metrics)

    return results_dict


def test_ablation(images, labels, batch_size, model, daug_params, repetitions,
                  layer_regex, ablation_pct, seed=None, metrics=None):
    """
    Tests the performance, as in test_rep(), of an ablated model.

    Parameters
    ----------
    images : h5py Dataset
        The set of images

    labels : h5py Dataset
        The ground truth labels

    batch_size : int
        Batch size

    model : Keras Model
        The model

    daug_params : dict
        Dictionary of data augmentation parameters

    repetitions : int
        Number of data augmentation repetitions

    Returns
    -------
    results_dict : dict
        Dictionary containing some performance metrics

    See
    ---
    test_rep()
    ablate_activations()
    """
    network_dict = network2dict(model)

    # Perform ablation (drop a set of the units)
    model_ablation = ablate_activations(model, layer_regex, ablation_pct, seed)
    
    results_dict = {}
    for r in range(repetitions):
        rep_dict =  test_rep(images, labels, batch_size, model_ablation,
                                 daug_params, 1, metrics)
        results_dict.update({r: rep_dict})
    results_dict = _stats_from_ablation_rep(results_dict)

    model = restore_nodes(model, network_dict)
    del model_ablation

    return results_dict


def test_adv(images, labels, batch_size, model, adv_model, daug_params, 
             attack_params):
    """
    Tests the performance of a model on adversarial images. The adversarial
    images are computed according to the attack specified in the arguments.

    Parameters
    ----------
    images : dask array
        The set of images

    labels : dask array
        The ground truth labels

    batch_size : int
        Batch size

    model : Keras Model
        The model

    adv_model : Keras Model
        The model used to generate adversarial examples

    daug_params : dict
        Dictionary of data augmentation parameters

    attack_params : dict
        Dictionary of the attack parameters

    Returns
    -------
    results_dict : dict
        Dictionary containing some performance metrics
    """

    # Get session
    sess = K.get_session()
    
    # Initialize adversarial attack
    attack, attack_params_cleverhans, bs = init_attack(
            adv_model, attack_params, sess)
    if bs:
        batch_size = bs

    n_images = images.shape[0]
    n_classes = labels.shape[1]
    n_batches_per_epoch = int(np.ceil(float(n_images) / batch_size))

    # Create batch generator
    image_gen = get_generator(images, **daug_params)
    batch_gen = batch_generator(image_gen, images, labels, batch_size, 
                                aug_per_im=1, shuffle=False)

    # Define input TF placeholder
    if daug_params['crop_size']:
        image_shape = daug_params['crop_size']
    else:
        image_shape = images.shape[1:]
    x = tf.placeholder(K.floatx(), shape=(bs,) + tuple(image_shape))
    y = tf.placeholder(K.floatx(), shape=(bs,) + (n_classes,))

    # Define adversarial predictions symbolically
    x_adv = attack.generate(x, **attack_params_cleverhans)
    x_adv = tf.stop_gradient(x_adv)
    predictions_adv = model(x_adv)

    # Define accuracy and mean squared error symbolically
    correct_preds = tf.equal(tf.argmax(y, axis=-1), 
                             tf.argmax(predictions_adv, axis=-1))
    acc_value = tf.reduce_mean(tf.to_float(correct_preds))
    mse_value = tf.reduce_mean(tf.square(tf.subtract(x, x_adv)))

    # Init results variables
    accuracy = 0.0
    mse = 0.0

    with sess.as_default():
        init = 0
        for _ in tqdm(range(n_batches_per_epoch)):
            batch = next(batch_gen())
            this_batch_size = batch[0].shape[0]

            # Evaluate accuracy
            if isinstance(batch[1], (list, )):
                yy = batch[1][0]
            else:
                yy = batch[1]

            # Evaluate accuracy and MSE
            batch_acc = acc_value.eval(feed_dict={x: batch[0], y: yy,
                                                  K.learning_phase(): 0})
            accuracy += (this_batch_size * batch_acc)
            batch_mse = mse_value.eval(feed_dict={x: batch[0],
                                       K.learning_phase(): 0})
            mse += (this_batch_size * batch_mse)

            init += this_batch_size

    accuracy /= n_images
    mse /= n_images

    results_dict = {'mean_acc': accuracy,
                    'mean_mse': mse}

    return results_dict


def activations_norm(images, labels, batch_size, model, layer_regex,
                     daug_params, norms=['fro']):
    """
    Computes the norm of the activation of all feature maps

    Parameters
    ----------
    images : h5py Dataset
        The set of images

    labels : h5py Dataset
        The ground truth labels

    batch_size : int
        Batch size

    model : Keras Model
        The model

    daug_params : dict
        Dictionary of data augmentation parameters

    Returns
    -------
    results_dict : dict
        Dictionary containing some performance metrics
    """
    def _update_stats(mean_norm, std_norm, norm):
        mean_norm_batch = np.mean(norm, axis=0)
        std_norm_batch = np.std(norm, axis=0)
        mean_norm = init / float(end) * mean_norm + \
                    batch_size / float(end) * mean_norm_batch
        std_norm = init / float(end) * std_norm ** 2 + \
                    batch_size / float(end) * std_norm_batch ** 2 + \
                    (init * batch_size) / float(end ** 2) * \
                    (mean_norm - mean_norm_batch) ** 2
        std_norm = np.sqrt(std_norm)

        return mean_norm, std_norm

    def _frobenius_norm(activations):
        norm = np.linalg.norm(
                activations, ord='fro', 
                axis=tuple(range(1, len(activations.shape) - 1)))
        return norm

    def _inf_norm(activations):
        norm = np.max(np.abs(activations),
                      axis=tuple(range(1, len(activations.shape) - 1)))
        return norm

    n_images = images.shape[0]
    n_batches_per_epoch = int(np.ceil(float(n_images) / batch_size))

    # Create batch generator
    image_gen = get_generator(images, **daug_params)
    batch_gen = batch_generator(image_gen, images, labels, batch_size, 
                                aug_per_im=1, shuffle=False)

    # Initialize list to store the mean norm of the activations
    results_dict = {'activations_norm': {}, 'summary': {}} 

    # Iterate over the layers
    model = del_extra_nodes(model)
    for layer in model.layers:
        if re.match(layer_regex, layer.name):
            layer_name = layer.name.encode('utf-8')
            print('\nLayer {}'.format(layer_name))
            output = model.get_layer(layer_name)\
                    .outbound_nodes[0].input_tensors[0]
            get_output = K.function([model.input, K.learning_phase()], 
                                    [output])
            n_channels = K.int_shape(output)[-1]
            results_dict['activations_norm'].update({layer_name: 
                {n: {'mean': np.zeros(n_channels), 
                     'std': np.zeros(n_channels)} for n in norms}})
            layer_dict = results_dict['activations_norm'][layer_name]
            init = 0
            batch_gen.image_gen.reset()
            for _ in tqdm(range(n_batches_per_epoch)):
                batch_images, _ = next(batch_gen())
                batch_size = batch_images.shape[0]
                end = init + batch_size
                activations = get_output([batch_images, 0])[0]
                for norm_key in norms:
                    mean_norm = layer_dict[norm_key]['mean']
                    std_norm = layer_dict[norm_key]['std']
                    if norm_key == 'fro':
                        norm = _frobenius_norm(activations)
                    elif norm_key == 'inf':
                        norm = _inf_norm(activations)
                    else:
                        raise NotImplementedError('Implemented norms are fro '
                                'and inf')
                    mean_norm, std_norm = _update_stats(mean_norm, std_norm, 
                                                        norm)
                    layer_dict[norm_key]['mean'] = mean_norm
                    layer_dict[norm_key]['std'] = std_norm
                init = end

    # Compute summary statistics across the channels
    for layer, layer_dict in results_dict['activations_norm'].items():
        results_dict['summary'].update({layer: {}})
        for norm_key, norm_dict in layer_dict.items():
            results_dict['summary'][layer].update({norm_key: {
                'mean': np.mean(norm_dict['mean']), 
                'std': np.mean(norm_dict['std'])}})

    return results_dict


def activations(images, labels, batch_size, model, layer_regex, nodaug_params, 
                daug_params, include_input=False, class_invariance=False, 
                n_daug_rep=0,  norms=['fro']):
    """
    Computes metrics from the activations, such as the norm of the feature
    maps, data augmentation invariance, class invariance, etc.

    Parameters
    ----------
    images : h5py Dataset
        The set of images

    labels : h5py Dataset
        The ground truth labels

    batch_size : int
        Batch size

    model : Keras Model
        The model

    nodaug_params : dict
        Dictionary of data augmentation parameters for the baseline

    daug_params : dict
        Dictionary of data augmentation parameters

    include_input : bool
        If True, the input layer is considered for the analysis

    class_invariance : bool
        If True, the class invariance score is computed

    n_daug_rep : int
        If larger than 0, the data augentation invariance score is computed,
        performing n_daug_rep repetitions of random augmentations

    norms : list
        List of keywords to specify the types of norms to compute on the 
        activations

    Returns
    -------
    results_dict : dict
        Dictionary containing some performance metrics
    """
    def _update_stats(mean_norm, std_norm, norm):
        mean_norm_batch = np.mean(norm, axis=0)
        std_norm_batch = np.std(norm, axis=0)
        mean_norm = init / float(end) * mean_norm + \
                    batch_size / float(end) * mean_norm_batch
        std_norm = init / float(end) * std_norm ** 2 + \
                    batch_size / float(end) * std_norm_batch ** 2 + \
                    (init * batch_size) / float(end ** 2) * \
                    (mean_norm - mean_norm_batch) ** 2
        std_norm = np.sqrt(std_norm)

        return mean_norm, std_norm

    def _frobenius_norm(activations):
        norm = np.linalg.norm(
                activations, ord='fro', 
                axis=tuple(range(1, len(activations.shape) - 1)))
        return norm

    def _inf_norm(activations):
        norm = np.max(np.abs(activations),
                      axis=tuple(range(1, len(activations.shape) - 1)))
        return norm

    model = del_extra_nodes(model)

    n_images = images.shape[0]
    n_batches_per_epoch = int(np.ceil(float(n_images) / batch_size))

    # Get relevant layers
    if include_input:
        layer_regex = '({}|.*input.*)'.format(layer_regex)
    else:
        layer_regex = layer_regex

    layers = [layer.name for layer in model.layers 
              if re.compile(layer_regex).match(layer.name)]

    # Initialize HDF5 to store the activations
#     filename = 'hdf5_aux_{}'.format(time.time())
#     activations_hdf5_aux = h5py.File(filename, 'w')
#     hdf5_aux = [filename]
# 
#     grp_activations = activations_hdf5_aux.create_group('activations')

    if class_invariance:
#         grp_labels = activations_hdf5_aux.create_group('labels')
        labels_true_da = []
        labels_pred_da = []
        predictions_da = []
#         labels_true = grp_labels.create_dataset(
#                 'labels_true', shape=(n_images, ), dtype=np.uint8)
#         labels_pred = grp_labels.create_dataset(
#                 'labels_pred', shape=(n_images, ), dtype=np.uint8)
#         predictions = grp_labels.create_dataset(
#                 'predictions', shape=labels.shape, dtype=K.floatx())
        idx_softmax = model.output_names.index('softmax')
        store_labels = True
    else:
        store_labels = False

    # Initialize results dictionary
    results_dict = {'activations_norm': {}, 'summary': {}, 
                    'class_invariance': {}, 'daug_invariance': {}} 

    # Iterate over the layers
    for layer_name in layers:

        # Create batch generator
        image_gen = get_generator(images, **nodaug_params)
        batch_gen = generate_batches(image_gen, images, labels, batch_size,
                                     aug_per_im=1, shuffle=False)

        layer = model.get_layer(layer_name)
        layer_shape = layer.output_shape[1:]
        n_channels = layer_shape[-1]

        if re.compile('.*input.*').match(layer_name):
            layer_name = 'input'

        print('\nLayer {}\n'.format(layer_name))

        # Create a Dataset for the activations of the layer
#         activations_layer = grp_activations.create_dataset(
#                 layer_name, shape=(n_images, ) + layer_shape, 
#                 dtype=K.floatx())
        # Create dask array for the activations of the layer
        activations_layer_da = []

        # Initialize placeholders in the results dict for the layer
        results_dict['activations_norm'].update({layer_name: 
            {n: {'mean': np.zeros(n_channels), 
                 'std': np.zeros(n_channels)} for n in norms}})
        layer_dict = results_dict['activations_norm'][layer_name]

        activation_function = K.function([model.input, 
                                          K.learning_phase()], 
                                         [layer.output])

        # Iterate over the data set in batches
        init = 0
        for batch_images, batch_labels in tqdm(
                batch_gen, total=n_batches_per_epoch):

            batch_size = batch_images.shape[0]
            end = init + batch_size

            # Store labels
            if store_labels:
                preds = model.predict_on_batch(batch_images)
                if isinstance(preds, list):
                    preds = preds[idx_softmax]
                labels_pred_da.append(da.from_array(
                    np.argmax(preds, axis=1)))
                labels_true_da.append(da.from_array(
                    np.argmax(batch_labels, axis=1)))
                predictions_da.append(da.from_array(preds))
#                 labels_pred[init:end] = np.argmax(preds, axis=1)
#                 labels_true[init:end] = np.argmax(batch_labels, axis=1)
#                 predictions[init:end, :] = preds

            # Get and store activations
            activations = activation_function([batch_images, 0])[0]
            activations_layer_da.append(da.from_array(
                activations, chunks=activations.shape))
#             activations_layer[init:end] = activations

            # Compute norms
            for norm_key in norms:
                mean_norm = layer_dict[norm_key]['mean']
                std_norm = layer_dict[norm_key]['std']
                if norm_key == 'fro':
                    norm = _frobenius_norm(activations)
                elif norm_key == 'inf':
                    norm = _inf_norm(activations)
                else:
                    raise NotImplementedError('Implemented norms are fro '
                            'and inf')
                mean_norm, std_norm = _update_stats(mean_norm, std_norm, 
                                                    norm)
                layer_dict[norm_key]['mean'] = mean_norm
                layer_dict[norm_key]['std'] = std_norm

            init = end
            if init == n_images:
                store_labels = False
                break

        # Concatenate dask arrays
        activations_layer_da = da.concatenate(activations_layer_da, axis=0)
        activations_layer_da = activations_layer_da.reshape((n_images, -1))
        d_activations = activations_layer_da.shape[-1]

        if class_invariance:
            print('\nComputing class invariance\n')
            labels_pred_da = da.concatenate(labels_pred_da)
            labels_true_da = da.concatenate(labels_true_da)
            predictions_da = da.concatenate(predictions_da)
            n_classes = len(np.unique(labels_true_da))

        # Compute MSE matrix of the activations
        r = da.reshape(da.sum(da.square(activations_layer_da), 
                                        axis=1), (-1, 1))
        mse_matrix_da = (r - 2 * da.dot(activations_layer_da,
                                     da.transpose(activations_layer_da)) \
                     + da.transpose(r)) / d_activations
        mse_matrix_da = mse_matrix_da.rechunk((mse_matrix_da.chunksize[0],
                                               mse_matrix_da.shape[-1]))

        # Compute class invariance
        time0 = time()
        results_dict['class_invariance'].update({layer_name: {}})
        class_invariance_scores_da = []
        if class_invariance:
#             mse_matrix_mean = da.mean(mse_matrix_da).compute()
            for cl in tqdm(range(n_classes)):
                labels_cl = labels_pred_da == cl
                labels_cl = labels_cl.compute()
                mse_class = mse_matrix_da[labels_cl, :][:, labels_cl]
                mse_class = mse_class.rechunk((-1, -1))
#                 mse_class_mean = da.mean(mse_class).compute()
#                 class_invariance_score = 1. - np.divide(
#                         mse_class_mean, mse_matrix_mean)
#                 results_dict['class_invariance'][layer_name].update(
#                         {cl: class_invariance_score})
                class_invariance_scores_da.append(
                        1. - da.divide(da.mean(mse_class),
                                       da.mean(mse_matrix_da)))

        # Compute data augmentation invariance
        print('\nComputing data augmentation invariance\n')
        mse_daug_da = []

        results_dict['daug_invariance'].update({layer_name: {}})

        for r in range(n_daug_rep):
            print('Repetition {}'.format(r))

            image_gen_daug = get_generator(images, **daug_params)
            batch_gen_daug = generate_batches(image_gen_daug, images, labels, 
                                              batch_size, aug_per_im=1, 
                                              shuffle=False)

            activations_layer_daug_da = []

            # Iterate over the data set in batches to compute activations
            init = 0
            for batch_images, batch_labels in tqdm(
                    batch_gen, total=n_batches_per_epoch):

                batch_size = batch_images.shape[0]
                end = init + batch_size

                # Get and store activations
                activations = activation_function([batch_images, 0])[0]
                activations_layer_daug_da.append(da.from_array(
                    activations, chunks=activations.shape))

                init = end
                if init == n_images:
                    break

            activations_layer_daug_da = da.concatenate(
                    activations_layer_daug_da, axis=0)
            activations_layer_daug_da = activations_layer_daug_da.reshape(
                    (n_images, -1))
            activations_layer_daug_da = activations_layer_daug_da.rechunk(
                    (activations_layer_daug_da.chunksize[0],
                     activations_layer_daug_da.shape[-1]))

            # Compute MSE daug
            mse_daug_da.append(da.mean(da.square(activations_layer_da - \
                                                 activations_layer_daug_da), 
                                       axis=1))

        mse_daug_da = da.stack(mse_daug_da, axis=1)

        mse_sum = da.repeat(da.reshape(da.sum(mse_matrix_da, axis=1),
                                       (n_images, 1)), n_daug_rep, axis=1)

        daug_invariance_score_da = 1 - n_images * da.divide(mse_daug_da, mse_sum)

        time1 = time()

        # Compute dask results and update results dict
        results_dask = da.compute(class_invariance_scores_da,
                                  daug_invariance_score_da)

        time2 = time()

        results_dict['class_invariance'][layer_name].update(
                {cl: cl_inv_score 
                    for cl, cl_inv_score in enumerate(results_dask[0])})
        results_dict['daug_invariance'].update({layer_name: 
            {r: daug_inv_score 
                for r, daug_inv_score in enumerate(results_dask[1].T)}})
    # Compute summary statistics of the norms across the channels
    for layer, layer_dict in results_dict['activations_norm'].items():
        results_dict['summary'].update({layer: {}})
        for norm_key, norm_dict in layer_dict.items():
            results_dict['summary'][layer].update({norm_key: {
                'mean': np.mean(norm_dict['mean']), 
                'std': np.mean(norm_dict['std'])}})

    return results_dict


def _stats_from_pred(predictions, labels, metrics):
    """
    Computes the accuracy of the mean and the median of a set of predictions, 
    obtained from performing random data augmentation. Besides the accuracy,
    additional metrics can be specified as an argument.

    Parameters
    ----------
    predictions : ndarray
        The predictions over the data set, with shape [n_data, n_classes, rep]

    labels : h5py Dataset
        The ground truth labels

    metrics : str list
        List of metrics to compute, besides the accuracy

    Returns
    -------
    results_dict : dict
        Dictionary containing the performance metrics
    """
    mean_predictions = np.mean(predictions, axis=2)
    median_predictions = np.median(predictions, axis=2)
    mean_std_predictions = np.mean(np.std(predictions, axis=2))

    # Create dictionary of results
    results_dict = {'mean_std_predictions': mean_std_predictions,
                    'metrics': {}}

    metric_funcs, metric_names = handle_metrics(metrics)
    for metric, metric_name in zip(metric_funcs, metric_names):
        mean_metric = np.mean(K.get_value(metric(labels,
                                                 mean_predictions)))
        median_metric = np.mean(K.get_value(metric(labels, 
                                                   median_predictions)))
        per_rep_metric = np.zeros(predictions.shape[-1])
        for r in range(predictions.shape[-1]):
            per_rep_metric[r] = np.mean(K.get_value(
                metric(labels, predictions[:, :, r])))

        # Update dict of results
        results_dict['metrics'].update({metric_name: {
            'mean': mean_metric,
            'median': median_metric,
            'per_rep': per_rep_metric}})

    return results_dict


def _stats_from_ablation_rep(input_dict):
    """
    Re-formats the results from a standard results dictionary into a more
    meaninful data structure for the ablation test.

    Parameters
    ----------
    results_dict : dict
        The original results dictionary

    Returns
    -------
    output_dict : dict
        The modified dictionary
    """
    output_dict = {'metrics' : {metric: {'per_rep': []} for metric in
            input_dict[0]['metrics'].keys()}}

    # Iterate over the repetitions
    for d in input_dict.values():
        for metric, data in d['metrics'].items():
            output_dict['metrics'][metric]['per_rep'].append(
                    data['per_rep'][0])

    # Compute mean and standard deviation
    for metric, data in output_dict['metrics'].items():
        output_dict['metrics'][metric].update(
                {'mean': np.mean(data['per_rep']),
                 'std': np.std(data['per_rep'])})

    return output_dict




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_file',
        type=str,
        default='/home/alex/git/research/projects/emotion/images/hdf5/'
                'cifar10.hdf5',
        help='Path to the HDF5 file containing the data set.'
    )
    parser.add_argument(
        '--group_tr',
        type=str,
        default=None,
        help='Group name in the HDF5 file indicating the train data set.'
    )
    parser.add_argument(
        '--group_tt',
        type=str,
        default=None,
        help='Group name in the HDF5 file indicating the test data set.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Number of images to process in a batch.'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help='Size of the dask array chunks'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model file (architecture + weights + optimizer state)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory where to write the output files. If -1, the directory'
             'of the model is used'
    )
    parser.add_argument(
        '--output_basename',
        type=str,
        default='test',
        help='Name, without extension, of the output file'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        dest='append',
        help='If True, the results are appended to the current results file'
    )
    parser.add_argument(
        '--test_config_file',
        type=str,
        default=None,
        help='Path to a test configuration file'
    )
    parser.add_argument(
        '--metrics',
        type=str,  # list of str
        nargs='*',  # 0 or more arguments can be given
        default=[],
        help='List of metrics to compute, separated by spaces'
    )
    parser.add_argument(
        '--orig',
        action='store_true',
        dest='orig',
        help='If True, the performance on the original images will be '
             'computed.'
    )
    parser.add_argument(
        '--daug_schemes',
        type=str,  # list of str
        nargs='*',  # 0 or more arguments can be given
        default=[],
        help='List of data augmentation schemes, e.g. light heavier'
    )
    parser.add_argument(
        '--daug_rep',
        type=int,
        default=5,
        help='Number of random repetitions for the test data augmentation'
    )
    parser.add_argument(
        '--ablation_pct',
        type=float, # list of float
        nargs='*', # 0 or more arguments can be given
        default=None,
        help='Percentage of activations to zero'
    )
    parser.add_argument(
        '--ablation_rep',
        type=int,
        default=None,
        help='Number of random repetitions for the unit ablation robustness'
    )
    parser.add_argument(
        '--ablation_seed',
        type=int,
        default=19,
        help='Seed for the Dropout layers to perform ablation studies'
    )
    parser.add_argument(
        '--adv_attacks',
        type=str,
        nargs='*', # 0 or more arguments can be given
        default=None,
        help='List of paths to the configuration files with the attack'
             'parameters'
    )
    parser.add_argument(
        '--adv_model',
        type=str,
        default=None,
        help='Model file (architecture + weights + optimizer state) used to '
             'generate the black box adversarial examples'
    )
    parser.add_argument(
        '--adv_pct_data',
        type=float,
        default=None,
        help='The percentage of the data to use to compute the adversarial '
             'robustness'
    )
    parser.add_argument(
        '--adv_shuffle_seed',
        type=int,
        default=None,
        help='Seed for shuffling the adversarial subset of data'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
