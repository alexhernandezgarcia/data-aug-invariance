"""
Routine for training a neural network with data augmentation and validation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.utils.generic_utils import Progbar
from keras.callbacks import ProgbarLogger
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.metrics import top_k_categorical_accuracy

from functools import partial, update_wrapper

import tensorflow as tf

import numpy as np
import h5py
import yaml

from data_input import dataset_characteristics, train_val_split
from data_input import validation_image_params, get_generator
from data_input import create_control_dataset
from data_input import generate_batches
import networks
from test import test
from utils import prepare_train_config, prepare_test_config
from utils import define_train_params
from utils import handle_metrics, change_metrics_names
from utils import dict2namespace, namespace2dict
from utils import get_daug_scheme_path
from utils import pairwise_loss, invariance_loss, weighted_loss, mean_loss
from utils import handle_train_dir
from utils import print_flags, write_flags, numpy_to_python
from utils import print_test_results, write_test_results
from utils import Gradients, RDM, PrintLearningRate, LossWeightsScheduler
from utils import write_tensorboard
from utils import sel_metrics
from utils import TrainingProgressLogger


import sys
import os
import argparse
import time

# Initialize the Flags container
FLAGS = None


def main(argv=None):

    handle_train_dir(FLAGS.train_dir)
    
    # Print and write the flag arguments
    print_flags(FLAGS)
    write_flags(FLAGS)

    K.set_floatx('float32')

    # Read or/and prepare train config dictionary
    if FLAGS.train_config_file:
        with open(FLAGS.train_config_file, 'r') as f_yml:
            train_config = yaml.load(f_yml)
    else:
        train_config = {}
    train_config = prepare_train_config(train_config, FLAGS)
    train_config = dict2namespace(train_config)

    # Set tensorflow and numpy seeds (weights initialization)
    if train_config.seeds.tf:
        tf.set_random_seed(train_config.seeds.tf)
    np.random.seed(train_config.seeds.np)

    # Open HDF5 file containing the data set
    hdf5_file = h5py.File(train_config.data.data_file, 'r')
    num_examples, num_classes, image_shape = dataset_characteristics(
            hdf5_file, train_config.data.group_tr, train_config.data.labels_id)
    train_config.data.n_classes = num_classes
    train_config.data.image_shape = image_shape

    # Determine the train and validation sets
    images_tr, images_val, labels_tr, labels_val, aux_hdf5 = \
            train_val_split(hdf5_file, 
                            train_config.data.group_tr,
                            train_config.data.group_val, 
                            train_config.data.chunk_size, 
                            train_config.data.pct_tr, 
                            train_config.data.pct_val, 
                            seed=train_config.seeds.train_val,
                            shuffle=train_config.data.shuffle_train_val,
                            labels_id=train_config.data.labels_id)
    train_config.data.n_train = images_tr.shape[0]
    train_config.data.n_val = images_val.shape[0]

    # Data augmentation parameters
    with open(train_config.daug.daug_params_file, 'r') as f_yml:
        daug_params_tr = yaml.load(f_yml)
        if (daug_params_tr['do_random_crop'] |
            daug_params_tr['do_central_crop']) & \
           (daug_params_tr['crop_size'] is not None):
            train_config.data.image_shape = daug_params_tr['crop_size']
    daug_params_tr['seed_daug'] = train_config.seeds.daug
    if train_config.daug.aug_per_img_val > 1:
        daug_params_val = daug_params_tr
        daug_params_val['seed_daug'] = train_config.seeds.daug
    else:
        daug_params_val = validation_image_params(
                train_config.daug.nodaug, **daug_params_tr)
    train_config.daug.daug_params_tr = daug_params_tr
    train_config.daug.daug_params_val = daug_params_val

    # Adjust training parameters
    train_config = define_train_params(train_config,
                                       output_dir=FLAGS.train_dir)

    # Read invariance paramters
    if train_config.optimizer.invariance:
        with open(train_config.optimizer.daug_invariance_params_file, 
                  'r') as f_yml:
            train_config.optimizer.daug_invariance_params = yaml.load(f_yml)
        with open(train_config.optimizer.class_invariance_params_file, 
                  'r') as f_yml:
            train_config.optimizer.class_invariance_params = yaml.load(f_yml)

    # Get monitored metrics
    metrics, metric_names = handle_metrics(train_config.metrics)
    FLAGS.metrics = metric_names

    # Initialize the model
    model, model_cat, loss_weights = _model_setup(
            train_config, metrics, FLAGS.resume_training)
    _model_print_save(model, FLAGS.train_dir)

    callbacks = _get_callbacks(train_config, FLAGS.train_dir,
                               save_model_every=FLAGS.save_model_every, 
                               track_gradients=FLAGS.track_gradients, 
                               fmri_rdms=FLAGS.fmri_rdms, 
                               loss_weights=loss_weights)

    # Write training configuration to disk
    output_file = os.path.join(FLAGS.train_dir, 'train_config_' +
                               time.strftime('%a_%d_%b_%Y_%H%M%S') + '.yml')
    with open(output_file, 'wb') as f:
        yaml.dump(numpy_to_python(namespace2dict(train_config)), f, 
                  default_flow_style=False)

    # Initialize Training Progress Logger
    loggers = []
    if FLAGS.log_file_train:
        log_file = os.path.join(FLAGS.train_dir, FLAGS.log_file_train)
        loggers.append(TrainingProgressLogger(log_file, model, train_config, 
                                              images_tr, labels_tr))
    if FLAGS.log_file_test:
        log_file = os.path.join(FLAGS.train_dir, FLAGS.log_file_test)
        loggers.append(TrainingProgressLogger(log_file, model, train_config, 
                                              images_val, labels_val))

    # Train
    history, model = train(images_tr, labels_tr, images_val, labels_val, model,
                           model_cat, callbacks, train_config, loggers)

    # Save model
    model.save(os.path.join(FLAGS.train_dir, 'model_final'))

    # Test
    if FLAGS.test_config_file:
        with open(FLAGS.test_config_file, 'r') as f_yml:
            test_config = yaml.load(f_yml)
        test_config = prepare_test_config(test_config, FLAGS)

        test_results_dict = test(images_val, labels_val, images_tr, labels_tr,
                                 model, test_config, 
                                 train_config.train.batch_size.val,
                                 train_config.data.chunk_size)

        # Write test results to YAML
        output_file = os.path.join(FLAGS.train_dir, 'test_' +
                                   os.path.basename(FLAGS.test_config_file))
        with open(output_file, 'wb') as f:
            yaml.dump(numpy_to_python(test_results_dict), f, 
                      default_flow_style=False)

        # Write test results to TXT
        output_file = output_file.replace('yml', 'txt')
        write_test_results(test_results_dict, output_file)

        # Print test results
        print_test_results(test_results_dict)


    # Close and remove aux HDF5 files
    hdf5_file.close()
    for f in aux_hdf5:
        filename = f.filename
        f.close()
        os.remove(filename)


def train(images_tr, labels_tr, images_val, labels_val, model, model_cat, 
          callbacks, train_config, loggers):

    # Create batch generators
    image_gen_tr = get_generator(images_tr, **train_config.daug.daug_params_tr)
    batch_gen_tr = generate_batches(
            image_gen_tr, images_tr, labels_tr,
            train_config.train.batch_size.gen_tr,
            aug_per_im=train_config.daug.aug_per_img_tr, shuffle=True,
            seed=train_config.seeds.batch_shuffle, 
            n_inv_layers=train_config.optimizer.n_inv_layers)
    image_gen_val = get_generator(images_val, 
                                  **train_config.daug.daug_params_val)
    batch_gen_val = generate_batches(
            image_gen_val, images_val, labels_val,
            train_config.train.batch_size.gen_val,
            aug_per_im=train_config.daug.aug_per_img_val, shuffle=False,
            n_inv_layers=train_config.optimizer.n_inv_layers)
    if FLAGS.no_val:
        batch_gen_val = None

    # Train model
    if FLAGS.no_fit_generator:

        metrics_names_val = ['val_{}'.format(metric_name) for metric_name in
                model.metrics_names]
        no_mean_metrics_progbar = True
#         no_mean_metrics_progbar = False

        for callback in callbacks.values():
            callback.set_model(model)
            callback.on_train_begin()

        for epoch in range(train_config.train.epochs):

            print('Epoch {}/{}'.format(epoch + 1, train_config.train.epochs))

            # Progress bar
    #         progbar = Progbar(target=train_config.train.batches_per_epoch_tr,
    #                           stateful_metrics=None)
            progbar = Progbar(target=train_config.train.batches_per_epoch_tr)

            for callback in callbacks.values():
                callback.on_epoch_begin(epoch)

            for batch_idx in range(train_config.train.batches_per_epoch_tr):

                for callback in callbacks.values():
                    callback.on_batch_begin(batch_idx)

                # Train
                batch = batch_gen_tr.next()
                debug = False

                # Log
                if loggers:
                    for logger in loggers:
                        logger.get_activations()

                # debug
                if debug:
                    preds = model.predict_on_batch(batch[0])
                    metrics = model.test_on_batch(batch[0], batch[1])
                    metrics_daug = metrics[model.metrics_names.index(
                        'daug_inv5_loss')]
                    metrics_class = metrics[model.metrics_names.index(
                        'class_inv5_loss')]
                    daug_true = batch[1][1]
                    daug_true_rel = daug_true[:, :, 0]
                    daug_true_all = daug_true[:, :, 1]
                    class_true = batch[1][2]
                    class_true_rel = class_true[:, :, 0]
                    class_true_all = class_true[:, :, 1]
                    pred_daug = preds[model.output_names.index(
                        'daug_inv5')][:, :, 0]
                    pred_class = preds[model.output_names.index(
                        'class_inv5')][:, :, 0]

                    num_daug = np.sum(daug_true_rel * pred_daug) / \
                               np.sum(daug_true_rel)
                    den_daug = np.sum(daug_true_all * pred_daug) / \
                               np.sum(daug_true_all)
                    loss_daug = num_daug / den_daug
                    num_class = np.sum(class_true_rel * pred_class) / \
                                np.sum(class_true_rel)
                    den_class = np.sum(class_true_all * pred_class) / \
                                np.sum(class_true_all)
                    loss_class = num_class / den_class
                # debug
                metrics = model.train_on_batch(batch[0], batch[1])
                if model_cat:
                    output_inv = model.predict_on_batch(batch[0])[0]
                    metrics_cat = model_cat.train_on_batch(
                            output_inv, batch[1][0])
                    metrics_names_cat = model_cat.metrics_names[:]
                else:
                    metrics_cat = []
                    metrics_names_cat = []

                # Progress bar
                if batch_idx + 1 < progbar.target:
                    metrics_progbar = sel_metrics(
                            model.metrics_names, metrics,
                            no_mean_metrics_progbar, 
                            metrics_cat=metrics_names_cat) 
                    metrics_progbar.extend(zip(metrics_names_cat,
                                               metrics_cat))
                    progbar.update(current=batch_idx + 1,
                                   values=metrics_progbar)

                # Log
                if loggers:
                    metrics_log = sel_metrics(
                            model.metrics_names, metrics,
                            no_mean=False, 
                            metrics_cat=metrics_names_cat) 
                    metrics_log.extend(zip(metrics_names_cat, metrics_cat))
                    for logger in loggers:
                        logger.log(metrics_log)

                for callback in callbacks.values():
                    callback.on_batch_end(batch_idx)

            # Validation
            metrics_val = np.zeros(len(metrics))
            for batch_idx in range(train_config.train.batches_per_epoch_val):

                batch = batch_gen_val.next()
                metrics_val_batch = model.test_on_batch(batch[0], batch[1])

                for idx, metric in enumerate(metrics_val_batch):
                    metrics_val[idx] += metric

            metrics_val /= train_config.train.batches_per_epoch_val
            metrics_val = metrics_val.tolist()

            # Progress bar
            metrics_progbar = sel_metrics(
                    model.metrics_names + metrics_names_val, 
                    metrics + metrics_val, no_mean_metrics_progbar, 
                    no_val_daug=train_config.daug.aug_per_img_val == 1)
            progbar.add(1, values=metrics_progbar)

            # Tensorboard
            metrics_names_tensorboard = progbar.sum_values.keys()
            metrics_tensorboard = [metric[0] / float(metric[1]) for metric in 
                    progbar.sum_values.values()]
            for metric_name, metric in zip(
                    model.metrics_names + metrics_names_val,
                    metrics + metrics_val):
                if metric_name not in metrics_names_tensorboard:
                    metrics_names_tensorboard.append(metric_name)
                    metrics_tensorboard.append(metric)
            metrics_tensorboard = sel_metrics(
                    metrics_names_tensorboard, 
                    metrics_tensorboard, no_mean=False, 
                    no_val_daug=train_config.daug.aug_per_img_val > 1,
                    metrics_cat=[])
            metrics_tensorboard = map(list, zip(*metrics_tensorboard))
            write_tensorboard(callbacks['tensorboard'], 
                              metrics_tensorboard[0], metrics_tensorboard[1], 
                              epoch)

            for callback in callbacks.values():
                callback.on_epoch_end(epoch)

        history = None
    else:
        history = model.fit_generator(
                generator=batch_gen_tr,
                steps_per_epoch=train_config.train.batches_per_epoch_tr,
                epochs=train_config.train.epochs,
                validation_data=batch_gen_val,
                validation_steps=train_config.train.batches_per_epoch_val,
                initial_epoch=train_config.train.initial_epoch,
                max_queue_size=train_config.data.queue_size,
                callbacks=callbacks.values())

    if loggers:
        for logger in loggers:
            logger.close()

    return history, model
    

def _model_setup(train_config, metrics, resume_training=None):

    if resume_training:
        model = load_model(os.path.join(resume_training))
        train_config.train.initial_epoch = int(resume_training.split('_')[-1])
    else:
        model = _model_init(train_config)
        train_config.train.initial_epoch = 0

    # Setup optimizer
    optimizer = _get_optimizer(train_config.optimizer,
                               train_config.train.lr.init_lr)
    optimizer_cat = _get_optimizer(train_config.optimizer,
                                   0.01)

    if isinstance(model, list):
        if train_config.optimizer.daug_invariance_params['pct_loss'] + \
           train_config.optimizer.class_invariance_params['pct_loss'] == 1.:
            model_cat = model[1]
            model_cat.compile(loss=train_config.optimizer.loss,
                              optimizer=optimizer_cat,
                              metrics=metrics)
            model = model[0]
        else: 
            model = model[0]
            model_cat = None
    else:
        model_cat = None

    # Get invariance layers
    inv_outputs = [output_name for output_name in model.output_names 
                   if '_inv' in output_name]
    daug_inv_outputs = [output_name for output_name in inv_outputs 
                   if 'daug_' in output_name]
    class_inv_outputs = [output_name for output_name in inv_outputs 
                   if 'class_' in output_name]
    mean_inv_outputs = [output_name for output_name in inv_outputs 
                   if 'mean_' in output_name]
    train_config.optimizer.n_inv_layers = len(daug_inv_outputs)

    if train_config.optimizer.invariance:
        # Determine loss weights for each invariance loss at each layer
        assert train_config.optimizer.daug_invariance_params['pct_loss'] +\
               train_config.optimizer.class_invariance_params['pct_loss'] \
               <= 1.
        no_inv_layers = []
        if FLAGS.no_inv_last_layer:
            no_inv_layers.append(len(daug_inv_outputs))
        if FLAGS.no_inv_first_layer:
            no_inv_layers.append(0)
        if FLAGS.no_inv_layers:
            no_inv_layers = [int(layer) - 1 for layer in FLAGS.no_inv_layers]
        daug_inv_loss_weights = get_invariance_loss_weights(
                train_config.optimizer.daug_invariance_params, 
                train_config.optimizer.n_inv_layers,
                no_inv_layers)
        class_inv_loss_weights = get_invariance_loss_weights(
                train_config.optimizer.class_invariance_params, 
                train_config.optimizer.n_inv_layers,
                no_inv_layers)
        mean_inv_loss_weights = np.zeros(len(mean_inv_outputs))
        loss_weight_cat = 1.0 - (np.sum(daug_inv_loss_weights) + \
                                 np.sum(class_inv_loss_weights))

        if 'decay_rate' in train_config.optimizer.daug_invariance_params or \
           'decay_rate' in train_config.optimizer.class_invariance_params:
            loss_weights_tensors = {'softmax': K.variable(loss_weight_cat,
                                                          name='w_softmax')}
            {loss_weights_tensors.update(
                {output: K.variable(weight, name='w_{}'.format(output))})
                for output, weight 
                in zip(daug_inv_outputs, daug_inv_loss_weights)}
            {loss_weights_tensors.update(
                {output: K.variable(weight, name='w_{}'.format(output))})
                for output, weight 
                in zip(class_inv_outputs, class_inv_loss_weights)}
            {loss_weights_tensors.update(
                {output: K.variable(weight, name='w_{}'.format(output))})
                for output, weight 
                in zip(mean_inv_outputs, mean_inv_loss_weights)}
            loss = {'softmax': weighted_loss(
                train_config.optimizer.loss, loss_weights_tensors['softmax'])}
            {loss.update({output: weighted_loss(
                invariance_loss, loss_weights_tensors[output])})
                for output in daug_inv_outputs}
            {loss.update({output: weighted_loss(
                invariance_loss, loss_weights_tensors[output])})
                for output in class_inv_outputs}
            {loss.update({output: weighted_loss(
                mean_loss, loss_weights_tensors[output])})
                for output in mean_inv_outputs}
            loss_weights = [1.] * len(model.outputs)
        else:
            loss = {'softmax': train_config.optimizer.loss}
            {loss.update({output: invariance_loss}) for output 
                    in daug_inv_outputs}
            {loss.update({output: invariance_loss}) for output 
                    in class_inv_outputs}
            {loss.update({output: mean_loss}) for output 
                    in mean_inv_outputs}
            if 'output_inv' in model.outputs:
                loss.update({'output_inv': None})
            loss_weights = {'softmax': loss_weight_cat}
            {loss_weights.update({output: loss_weight})
                for output, loss_weight in zip(daug_inv_outputs, 
                                           daug_inv_loss_weights)}
            {loss_weights.update({output: loss_weight})
                for output, loss_weight in zip(class_inv_outputs, 
                                           class_inv_loss_weights)}
            {loss_weights.update({output: loss_weight})
                for output, loss_weight in zip(mean_inv_outputs, 
                                           mean_inv_loss_weights)}
            loss_weights_tensors = None

        metrics_dict = {'softmax': metrics}
        model.compile(loss=loss,
                      loss_weights=loss_weights,
                      optimizer=optimizer,
                      metrics=metrics_dict)
    else:
        model.compile(loss=train_config.optimizer.loss,
                      optimizer=optimizer,
                      metrics=metrics)
        loss_weights_tensors = None

    # Change metrics names
    model = change_metrics_names(model, train_config.optimizer.invariance)

    if model_cat:
        model_cat = change_metrics_names(model_cat, False)

    return model, model_cat, loss_weights_tensors


def _model_init(train_config):
    if train_config.network.name == 'allcnn':
        model = networks.allcnn(
                    train_config.data.image_shape, 
                    train_config.data.n_classes,
                    dropout=train_config.network.reg.dropout,
                    weight_decay=train_config.network.reg.weight_decay,
                    batch_norm=train_config.network.batch_norm,
                    depth=train_config.network.depth,
                    id_output=train_config.optimizer.invariance)
    elif train_config.network.name == 'allcnn_large':
        model = networks.allcnn_large(
                    train_config.data.image_shape, 
                    train_config.data.n_classes,
                    dropout=train_config.network.reg.dropout,
                    weight_decay=train_config.network.reg.weight_decay,
                    batch_norm=train_config.network.batch_norm,
                    depth=train_config.network.depth,
                    id_output=train_config.optimizer.invariance,
                    stride_conv1=train_config.network.stride_conv1)
    elif train_config.network.name == 'allcnn_mnist':
        model = networks.allcnn_mnist(
                    train_config.data.image_shape, 
                    train_config.data.n_classes,
                    dropout=train_config.network.reg.dropout,
                    weight_decay=train_config.network.reg.weight_decay,
                    batch_norm=train_config.network.batch_norm,
                    depth=train_config.network.depth,
                    id_output=train_config.optimizer.invariance)
    elif train_config.network.name == 'wrn':
        model = networks.wrn(
                    train_config.data.image_shape, 
                    train_config.data.n_classes,
                    dropout=train_config.network.reg.dropout,
                    weight_decay=train_config.network.reg.weight_decay,
                    batch_norm=train_config.network.batch_norm,
                    blocks_per_group=train_config.network.blocks_per_group,
                    widening_factor=train_config.network.widening_factor,
                    id_output=train_config.optimizer.invariance)
    elif train_config.network.name == 'wrn_imagenet':
        model = networks.wrn(
                    train_config.data.image_shape, 
                    train_config.data.n_classes,
                    dropout=train_config.network.reg.dropout,
                    weight_decay=train_config.network.reg.weight_decay,
                    batch_norm=train_config.network.batch_norm,
                    blocks_per_group=train_config.network.blocks_per_group,
                    widening_factor=train_config.network.widening_factor,
                    stride_conv1=2)
    elif train_config.network.name == 'densenet':
        model = networks.densenet(
                    train_config.data.image_shape, 
                    train_config.data.n_classes,
                    train_config.network.blocks,
                    train_config.network.growth_rate,
                    train_config.network.theta,
                    dropout=train_config.network.reg.dropout,
                    weight_decay=train_config.network.reg.weight_decay,
                    batch_norm=train_config.network.batch_norm,
                    id_output=train_config.optimizer.invariance)
    elif train_config.network.name == 'lenet':
        model = networks.lenet(
                    train_config.data.image_shape, 
                    train_config.data.n_classes,
                    dropout=train_config.network.reg.dropout,
                    weight_decay=train_config.network.reg.weight_decay,
                    batch_norm=train_config.network.batch_norm,
                    id_output=train_config.optimizer.invariance)
    else:
        raise(NotImplementedError('Only networks implemented are allcnn'
                                  'and wrn'))

    return model        


def _model_print_save(model, output_dir):
    # Print the model summary
    model.summary()

    # Save the model summary as a text file
    output_file = os.path.join(output_dir, 'arch_' +
                               time.strftime('%a_%d_%b_%Y_%H%M%S') + '.txt')
    with open(output_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save the model as a YAML file
    model_yaml = model.to_yaml()
    output_file = os.path.join(output_dir, 'arch_' +
                               time.strftime('%a_%d_%b_%Y_%H%M%S') + '.yml')
    with open(output_file, 'w') as f:
        f.write(model_yaml)



def _get_optimizer(optimizer_params, init_lr):
    if optimizer_params.name.lower() == 'sgd':
        optimizer = SGD(lr=init_lr,
                        momentum=optimizer_params.momentum,
                        nesterov=optimizer_params.nesterov)
    elif optimizer_params.name.lower() == 'adam':
        optimizer = Adam(init_lr)
    else:
        raise(NotImplementedError('Valid optimizers are: SGD and Adam'))

    return optimizer


def _get_callbacks(train_config, output_dir, save_model_every,
                   track_gradients=False, fmri_rdms=None, 
                   stateful_metrics=None, loss_weights=None):
    callbacks = {}

    # Callback: decay of learning rate
    if 'decay_factor' in train_config.train.lr and \
       'decay_epochs' in train_config.train.lr:
        if train_config.train.epochs != train_config.train.epochs_orig:
            mult = float(train_config.train.epochs) / \
                   train_config.train.epochs_orig
            train_config.train.lr.decay_epochs = [int(d * mult) 
                    for d in train_config.train.lr.decay_epochs]
        lr_decay_schedule = lr_decay(train_config.train.lr.init_lr,
                                     train_config.train.lr.decay_factor,
                                     train_config.train.lr.decay_epochs)
        callback_lr_decay = LearningRateScheduler(lr_decay_schedule)
        callbacks.update({'lr_decay': callback_lr_decay})

    # Callback: TensorBoard
    callback_tensorboard = TensorBoard(log_dir=output_dir)
    callbacks.update({'tensorboard': callback_tensorboard})

    # Callback: Save model
    if save_model_every > 0:
        model_filename = os.path.join(output_dir, 'model_' +
                                      time.strftime('%a_%d_%b_%Y_%H%M%S') +
                                      '_{epoch:03d}')
        callback_model_ckpt = ModelCheckpoint(filepath=model_filename,
                                              period=save_model_every)
        callbacks.update({'model_ckpt': callback_model_ckpt})

    # Callback: Gradients
    if track_gradients:
        callback_gradients = Gradients()
        callbacks.update({'gradients': callback_gradients})

    # Callback: RDM
    if fmri_rdms:
        callback_rdm = RDM(train_dir=output_dir,
                           hdf5_file=fmri_rdms)
        callbacks.update({'rdm': callback_rdm})

    # Callback: Progress Bar
    if stateful_metrics:
        callback_progbar = ProgbarLogger(stateful_metrics=stateful_metrics)
        callbacks.update({'progbar': callback_progbar})

    # Callback: Loss Weight Scheduler
    if loss_weights:
        daug_inv_params = train_config.optimizer.daug_invariance_params
        if 'decay_rate' in daug_inv_params and \
           'decay_epochs_pct' in daug_inv_params:
            decay_epochs_daug = train_config.train.epochs * \
                                np.asarray(
                                        daug_inv_params['decay_epochs_pct'])
            decay_epochs_daug = decay_epochs_daug.astype(int).tolist()
            decay_rate_daug = daug_inv_params['decay_rate']
        else:
            decay_epochs_daug = []
            decay_rate_daug = 1.
        class_inv_params = train_config.optimizer.class_invariance_params
        if 'decay_rate' in class_inv_params and \
           'decay_epochs_pct' in class_inv_params:
            decay_epochs_class = train_config.train.epochs * \
                                np.asarray(
                                        class_inv_params['decay_epochs_pct'])
            decay_epochs_class = decay_epochs_class.astype(int).tolist()
            decay_rate_class = class_inv_params['decay_rate']
        else:
            decay_epochs_class = []
            decay_rate_class = 1.
        callback_loss_weights = LossWeightsScheduler(
                loss_weights=loss_weights,
                decay_epochs_daug=decay_epochs_daug,
                decay_epochs_class=decay_epochs_class,
                decay_rate_daug=decay_rate_daug,
                decay_rate_class=decay_rate_class)
        callbacks.update({'loss_weights': callback_loss_weights})

    return callbacks


def lr_decay(initial_lr, lr_decay_factor, key_epochs):
    """
    Function to receive the parameters of decay schedule

    Parameters
    ----------
    initial_lr : float
        Initial learning rate

    lr_decay_factor : float
        Learning rate decay factor

    key_epochs : list
        Epochs at which the learning rate is decayed.

    Returns
    -------
    lr_decay_schedule : function
        Function that gets only as parameter the current epoch
    """

    def lr_decay_schedule(epoch):
        """
        Defines a learning rate decay schedule schedule as a function of the
        current epoch.

        Parameters
        ----------
        epoch : int
            The current epoch

        Returns
        -------
        lr : float
            The learning rate as a function of the current epoch.
        """

        step = 0
        for e in key_epochs:
            if epoch < e:
                break
            else:
                step += 1
        lr = initial_lr * lr_decay_factor ** step

        return lr

    return lr_decay_schedule


def get_invariance_loss_weights(invariance_params, n_inv_layers, 
                                zero_loss_layers=[]):
    """
    Determines the weight of the loss function for each invariance layer, 
    according to the parameters file

    Parameters
    ----------
    invariance_params : dict
        Dictionary of image id parameters

    n_inv_layers : int
        Number of invariance layers in the architecture

    zero_loss_layers : list
        Indices of the layers that should be assigned 0. weight

    Returns
    -------
    inv_loss_weights : float list
        List of weights for the loss function, per invariance layer
    """
    inv_loss_weights_final = np.zeros(n_inv_layers)
    inv_layers = [idx for idx in range(n_inv_layers) if idx not in
            zero_loss_layers]
    n_inv_layers = n_inv_layers - len(zero_loss_layers)

    if invariance_params['distr'] == 'zeros':
        inv_loss_weights = np.zeros(n_inv_layers, dtype=float)
    elif invariance_params['distr'] == 'uniform':
        inv_loss_weights = n_inv_layers * \
                     [invariance_params['pct_loss'] / n_inv_layers]
    elif invariance_params['distr'] == 'linear':
        inv_loss_weights = np.linspace(start=1., 
                                       stop=invariance_params['diff_max_min'], 
                                       num=n_inv_layers)
        inv_loss_weights /= np.sum(inv_loss_weights)
        inv_loss_weights *= invariance_params['pct_loss']
    elif invariance_params['distr'] == 'exponential':
        inv_loss_weights = np.logspace(start=0., 
                                       stop=1.,
                                       base=invariance_params['diff_max_min'],
                                       num=n_inv_layers)
        inv_loss_weights /= np.sum(inv_loss_weights)
        inv_loss_weights *= invariance_params['pct_loss']
    else:
        raise NotImplementedError('The distribution can be uniform, linear or '
                                  'exponential')

    for idx, weight in zip(inv_layers, inv_loss_weights):
        inv_loss_weights_final[idx] = weight

    return inv_loss_weights_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_config_file',
        type=str,
        default='config.yml',
        help='Path to the training configuration file'
    )
    parser.add_argument(
        '--test_config_file',
        type=str,
        default=None,
        help='Path to the test configuration file'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help='Path to the HDF5 file containing the data set.'
    )
    parser.add_argument(
        '--group_tr',
        type=str,
        default=None,
        help='Group name in the HDF5 file indicating the train data set.'
    )
    parser.add_argument(
        '--group_val',
        type=str,
        default=None,
        help='Group name in the HDF5 file indicating the test data set.'
    )
    parser.add_argument(
        '--labels_id',
        type=str,
        default='labels',
        help='String name of the h5py Dataset containing the labels'
    )
    parser.add_argument(
        '--shuffle_train_val',
        action='store_true',
        dest='shuffle_train_val',
        help='If true, the data samples will be shuffled before creating the '
             'training and validation partitions. Only relevant if no '
             'validation group is specified'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/dreamlearning/net2conv2fc_train',
        help='Directory where to write event logs and checkpoint'
    )
    parser.add_argument(
        '--pct_val',
        type=float,
        default=0.2,
        help='Percentage of samples for the validation set'
    )
    parser.add_argument(
        '--pct_tr',
        type=float,
        default=1.0,
        help='Percentage of examples to use from the training set.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Train batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--network_name',
        type=str,
        default=None,
        help='Identifier name of the network architecture'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=None,
        help='Hyperparameter of the weight decay regularization'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Add dropout regularization to the model.'
    )
    parser.add_argument(
        '--no_dropout',
        action='store_true',
        dest='no_dropout',
        help='Remove dropout regularization from to the model.'
    )
    parser.add_argument(
        '--batch_norm',
        action='store_true',
        dest='batch_norm',
        help='Add batch normalization to the model.'
    )
    parser.add_argument(
        '--no_batch_norm',
        action='store_true',
        dest='no_batch_norm',
        help='Remove batch normalization from to the model.'
    )
    parser.add_argument(
        '--metrics',
        type=str,  # list of str
        nargs='*',  # 0 or more arguments can be given
        default=None,
        help='List of metrics to monitor, separated by spaces'
    )
    parser.add_argument(
        '--daug_invariance_params',
        type=str,
        default=None,
        help='Path to configuration file with the data augmentation '
        'invariance parameters'
    )
    parser.add_argument(
        '--class_invariance_params',
        type=str,
        default=None,
        help='Path to configuration file with the class invariance parameters'
    )
    parser.add_argument(
        '--daug_params',
        type=str,
        default= None,
        help='Base name of the configuration file with the data augmentation '
             'parameters. It is expected to be located in '
             './daug_schemes/<dataset>/'
    )
    parser.add_argument(
        '--attack_params_file',
        type=str,
        default="attacks/fgsm_eps03.yml",
        help='Path to the configuration file with the attack parameters'
    )
    parser.add_argument(
        '--model_adv',
        type=str,
        default=None,
        help='Model file (architecture + weights + optimizer state) used to '
             'generate the adversarial examples'
    )
    parser.add_argument(
        '--save_model_every',
        type=int,
        default=50,
        help='Specifies epochs interval to save the model'
             '(architecture + weights + optimizer state)'
    )
    parser.add_argument(
        '--save_final_model',
        action='store_true',
        dest='save_final_model',
        help='Save the model (architecture + weights + optimizer state) at the'
             'end of the training process.'
    )
    parser.add_argument(
        '--track_gradients',
        action='store_true',
        dest='track_gradients',
        help='Enable callback to track gradients after every weight update'
    )
    parser.add_argument(
        '--fmri_rdms',
        type=str,
        default=None,
        help='HDF5 file containing the fMRI dataset'
    )
    parser.add_argument(
        '--resume_training',
        type=str,
        default=None,
        help='Model file (architecture + weights + optimizer state) to load'
             'and use to resume the training process'
    )
    parser.add_argument(
        '--aug_per_img_tr',
        type=int,
        default=None,
        help='Number of augmentation per image, per batch, for training'
    )
    parser.add_argument(
        '--aug_per_img_val',
        type=int,
        default=None,
        help='Number of augmentation per image in the validation set.'
    )
    parser.add_argument(
        '--no_inv_last_layer',
        action='store_true',
        dest='no_inv_last_layer',
        help='If True, the loss weight for the invariance term corresponding '
             'to the last layer will be zero'
    )
    parser.add_argument(
        '--no_inv_first_layer',
        action='store_true',
        dest='no_inv_first_layer',
        help='If True, the loss weight for the invariance term corresponding '
             'to the first layer will be zero'
    )
    parser.add_argument(
        '--no_inv_layers',
        type=str,  # list of str
        nargs='*',  # 0 or more arguments can be given
        default=None,
        help='List of layers (int, from 1) whose invariance loss should be 0'
    )
    parser.add_argument(
        '--simulate_norep_samples',
        action='store_true',
        dest='simulate_norep_samples',
        help='If true, either the batch size is multiplied or the learning '
             'rate divided by aug_per_im and the to compensate for the '
             'repeated samples within a batch'
    )
    parser.add_argument(
        '--simulate_rep_samples',
        type=int,
        default=None,
        help='If not None, either the batch size is divided or the learning '
             'rate multiplied by simulate_rep_samples in order to simulate '
             'repeated samples within a batch'
    )
    parser.add_argument(
        '--simulate_bs_lr',
        type=str,
        default='',
        help='Determines whether the simulation of rep/norep samples is done '
             'by adapting the learning rate or the batch size'
    )
    parser.add_argument(
        '--true_epochs',
        action='store_true',
        dest='true_epochs',
        help='If True, the number of batches per epoch is computed according '
             'to the number of different samples within the batches'
    )
    parser.add_argument(
        '--seed_tf',
        type=int,
        default=None,
        help='Random seed for TensorFlow'
    )
    parser.add_argument(
        '--seed_np',
        type=int,
        default=None,
        help='Random seed for Numpy'
    )
    parser.add_argument(
        '--seed_daug',
        type=int,
        default=None,
        help='Random seed for the flow() function of the image generator'
    )
    parser.add_argument(
        '--seed_batch_shuffle',
        type=int,
        default=None,
        help='Random seed for shuffling the batches'
    )
    parser.add_argument(
        '--seed_train_val',
        type=int,
        default=None,
        help='Random seed for the train-validation sets splitting'
    )
    parser.add_argument(
        '--test_rep',
        type=int,
        default=0,
        help='Number of repetitions of evaluation with random data'
             'augmentation on the test set'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help='Size of the dask array chunks'
    )
    parser.add_argument(
        '--queue_size',
        type=int,
        default=10,
        help='Size of the queue of the generator'
    )
    parser.add_argument(
        '--log_file_test',
        type=str,
        default='',
        help='CSV file name for the logger of test data'
    )
    parser.add_argument(
        '--log_file_train',
        type=str,
        default='',
        help='CSV file name for the logger of train data'
    )
    parser.add_argument(
        '--no_fit_generator',
        action='store_true',
        dest='no_fit_generator',
        help='Train with train_on_batch(), instead of fit_generator()'
    )
    parser.add_argument(
        '--force_invariance',
        action='store_true',
        dest='force_invariance',
        help='Force invariance training despite aug_per_img = 0'
    )
    parser.add_argument(
        '--no_val',
        action='store_true',
        dest='no_val',
        help='Disable validation after each epoch'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
