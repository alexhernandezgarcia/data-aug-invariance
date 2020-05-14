"""
Routine for computing the layer-wise data augmentation invariance of a model's
activations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import yaml

from scipy.io import savemat

import h5py
import pickle
import dask.array as da
import dask
from dask.diagnostics import ProgressBar
# See: https://docs.dask.org/en/latest/diagnostics-local.html

# Dask distributed
# See: https://docs.dask.org/en/latest/scheduling.html
# See: https://docs.dask.org/en/latest/setup/single-distributed.html
from dask.distributed import Client, LocalCluster

from data_input import hdf52dask, get_generator, batch_generator
from utils import get_daug_scheme_path
from utils import pairwise_loss, mean_loss, invariance_loss
from activations import get_activations

import keras.backend as K
from keras.models import load_model
import keras.losses
keras.losses.pairwise_loss = pairwise_loss
keras.losses.invariance_loss = invariance_loss
keras.losses.mean_loss = mean_loss

import os
import argparse
import shutil
from tqdm import tqdm, trange
from time import time

import re

# Initialize the Flags container
FLAGS = None


def main(argv=None):

#     cluster = LocalCluster(dashboard_address=None)
#     client = Client(cluster, memory_limit='{}GB'.format(FLAGS.memory_limit),
#                     processes=False)

    K.set_floatx('float32')

    chunk_size = FLAGS.chunk_size

    # Read data set
    hdf5_file = h5py.File(FLAGS.data_file, 'r')
    images, labels, _ = hdf52dask(hdf5_file, FLAGS.group, chunk_size,
                                  shuffle=FLAGS.shuffle, seed=FLAGS.seed, 
                                  pct=FLAGS.pct)
    n_images = images.shape[0]
    n_batches = int(np.ceil(n_images / float(FLAGS.batch_size)))

    # Data augmentation parameters
    daug_params_file = get_daug_scheme_path(FLAGS.daug_params, FLAGS.data_file)
    daug_params = yaml.load(open(daug_params_file, 'r'))
    nodaug_params_file = get_daug_scheme_path('nodaug.yml', FLAGS.data_file)
    nodaug_params = yaml.load(open(nodaug_params_file, 'r'))

    # Initialize the network model
    model_filename = FLAGS.model
    model = load_model(model_filename)

    # Print the model summary
    model.summary()

    # Get relevant layers
    if FLAGS.store_input:
        layer_regex = '({}|.*input.*)'.format(FLAGS.layer_regex)
    else:
        layer_regex = FLAGS.layer_regex

    layers = [layer.name for layer in model.layers 
              if re.compile(layer_regex).match(layer.name)]

    # Create batch generators
    n_daug_rep = FLAGS.n_daug_rep
    n_diff_per_batch = int(FLAGS.batch_size / n_daug_rep)
    image_gen_daug = get_generator(images, **daug_params)
    batch_gen_daug = batch_generator(image_gen_daug, images, labels, 
                                     batch_size=n_diff_per_batch, 
                                     aug_per_im=n_daug_rep, 
                                     shuffle=False)
    image_gen_nodaug = get_generator(images, **nodaug_params)
    batch_gen_nodaug = batch_generator(image_gen_nodaug, images, labels, 
                                        FLAGS.batch_size, aug_per_im=1, 
                                        shuffle=False)

    # Outputs
    if FLAGS.output_dir == '-1':
        FLAGS.output_dir = os.path.dirname(FLAGS.model)

    output_hdf5 = h5py.File(os.path.join(
        FLAGS.output_dir, FLAGS.output_mse_matrix_hdf5), 'w')
    output_pickle = os.path.join(FLAGS.output_dir, FLAGS.output_pickle)
    df_init_idx = 0
    df = pd.DataFrame()

    # Iterate over the layers
    for layer_idx, layer_name in enumerate(layers):

        # Reload the model
        if layer_idx > 0:
            K.clear_session()
            model = load_model(model_filename)

        layer = model.get_layer(layer_name)

        # Rename input layer
        if re.compile('.*input.*').match(layer_name):
            layer_name = 'input'

        hdf5_layer = output_hdf5.create_group(layer_name)

        activation_function = K.function([model.input, 
                                          K.learning_phase()], 
                                         [layer.output])

        print('\nComputing pairwise similarity at layer {}'.format(layer_name))

        # Compute activations of original data (without augmentation)
        a_nodaug_da = get_activations(activation_function, batch_gen_nodaug)
        a_nodaug_da = da.squeeze(a_nodaug_da)
        a_nodaug_da = da.rechunk(a_nodaug_da, 
                                 (chunk_size, ) + (a_nodaug_da.shape[1:]))
        dim_activations = a_nodaug_da.shape[1]

        # Comute matrix of similarities
        r = da.reshape(da.sum(da.square(a_nodaug_da), axis=1), (-1, 1))
        mse_matrix = (r - 2 * da.dot(a_nodaug_da,
                                     da.transpose(a_nodaug_da)) \
                     + da.transpose(r)) / dim_activations

        # Compute activations with augmentation
        a_daug_da = get_activations(activation_function, batch_gen_daug)
        a_daug_da = da.rechunk(a_daug_da, 
                               (chunk_size, dim_activations, 1))

        # Compute similarity of augmentations with respect to the
        # activations of the original data
        a_nodaug_da = da.repeat(da.reshape(a_nodaug_da, 
                                        a_nodaug_da.shape + (1, )), 
                             repeats=n_daug_rep, axis=2)
        a_nodaug_da = da.rechunk(a_nodaug_da, 
                               (chunk_size, dim_activations, 1))
        mse_daug = da.mean(da.square(a_nodaug_da - a_daug_da), axis=1)

        # Compute invariance score
        mse_sum = da.repeat(da.reshape(da.sum(mse_matrix, axis=1),
                                       (n_images, 1)), 
                            repeats=n_daug_rep, axis=1)
        mse_sum = da.rechunk(mse_sum, (chunk_size, 1))
        invariance = 1 - n_images * da.divide(mse_daug, mse_sum)

        print('Dimensionality activations: {}x{}x{}'.format(
            n_images, dim_activations, n_daug_rep))

        # Store HDF5 file
        if FLAGS.output_mse_matrix_hdf5:
            mse_matrix_ds = hdf5_layer.create_dataset(
                    'mse_matrix', shape=mse_matrix.shape, 
                    chunks=mse_matrix.chunksize, dtype=K.floatx())
            mse_daug_ds = hdf5_layer.create_dataset(
                    'mse_daug', shape=mse_daug.shape, 
                    chunks=mse_daug.chunksize, dtype=K.floatx())
            invariance_ds = hdf5_layer.create_dataset(
                    'invariance', shape=invariance.shape, 
                    chunks=invariance.chunksize, dtype=K.floatx())
            time_init = time()
            with ProgressBar(dt=1):
                da.store([mse_matrix, mse_daug, invariance], 
                         [mse_matrix_ds, mse_daug_ds, invariance_ds])
            time_end = time()
            print('Elapsed time: {}'.format(time_end - time_init))

            invariance = np.ravel(np.asarray(
                output_hdf5[layer_name]['invariance']))
        else:
            time_init = time()
            invariance = da.ravel(invariance).compute()
            time_end = time()
            print('Elapsed time: {}'.format(time_end - time_init))

        # Update pandas data frame for plotting
        df_end_idx = df_init_idx + n_images * n_daug_rep
        d = pd.DataFrame({'Layer': layer_name,
                          'sample': np.repeat(np.arange(n_images), n_daug_rep),
                          'n_daug': np.tile(np.arange(n_daug_rep), n_images),
                          'invariance': invariance}, 
                          index=np.arange(df_init_idx, df_end_idx).tolist())
        df = df.append(d)
        df_init_idx += df_end_idx

    pickle.dump(df, open(output_pickle, 'wb'))
    output_hdf5.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_file',
        type=str,
        default='/mnt/data/alex/datasets/hdf5/fmri_images.hdf5'
                'cifar10.hdf5',
        help='Path to the HDF5 file containing the data set.'
    )
    parser.add_argument(
        '--group',
        type=str,
        default='fmri',
        help='Group name in the HDF5 file indicating the train data set.'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help='Size of the dask array chunks'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for iterating over the data set'
    )
    parser.add_argument(
        '--pct',
        type=float,
        default=1.,
        help='Percentage of the data set to use'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        dest='shuffle',
        help='Whether to shuffle the samples, if pct is less than 1'
    )
    parser.add_argument(
        '--seed',
        type=int,
        dest='seed',
        help='Random seed for the data shuffling'
    )
    parser.add_argument(
        '--daug_params',
        type=str,
        default='nodaug.yml',
        help='Base name of the configuration file with the data augmentation '
             'parameters. It is expected to be located in '
             './daug_schemes/<dataset>/'
    )
    parser.add_argument(
        '--n_daug_rep',
        type=int,
        default=1,
        help='The number of HDF5 files with activations from data augmentation'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='-1',
        help='Directory where to write the output files. If -1, the directory'
             'of the model is used'
    )
    parser.add_argument(
        '--output_mse_matrix_hdf5',
        type=str,
        default=None,
        help='Output HDF5 file'
    )
    parser.add_argument(
        '--output_pickle',
        type=str,
        default=None,
        help='Output pickled file containing the invariance scores'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model file (architecture + weights + optimizer state) to load'
    )
    parser.add_argument(
        '--layer_regex',
        type=str,
        default='g[0-9]b[0-9]add',
        help='Regular expression of the name of the layer at which the '
             'activations will be computed'
    )
    parser.add_argument(
        '--memory_limit',
        type=int,
        default=128,
        help='Memory limit for the dask client [GB]'
    )
    parser.add_argument(
        '--store_input',
        action='store_true',
        dest='store_input',
        help='If True, store the input data'
    )
    parser.add_argument(
        '--store_labels',
        action='store_true',
        dest='store_labels',
        help='If True, store the labels and the predictions'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
