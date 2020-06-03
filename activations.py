"""
Library of functions related to the intermediate activations of DNN models
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
# from dask.diagnostics import ProgressBar
# See: https://docs.dask.org/en/latest/diagnostics-local.html

# Dask distributed
# See: https://docs.dask.org/en/latest/scheduling.html
# See: https://docs.dask.org/en/latest/setup/single-distributed.html
# from dask.distributed import Client, LocalCluster

from data_input import hdf52dask, get_generator
from data_input import batch_generator

import tensorflow.compat.v1.keras.backend as K

import os
import argparse
import shutil
from tqdm import tqdm, trange
from time import time

import re

# Initialize the Flags container
FLAGS = None


def get_activations(activation_function, batch_gen):
    """
    Computes the activations of a data set at one layer of the model in a 
    "delayed" way (for memory and computation efficiency) and return them as a
    dask array. 

    See: https://docs.dask.org/en/latest/delayed.html
    """

    layer_shape = K.int_shape(activation_function.outputs[0])[1:]
    layer_dim = np.prod(K.int_shape(activation_function.outputs[0])[1:])
    n_images = batch_gen.n_images
    n_aug = batch_gen.aug_per_im
    batch_size = batch_gen.batch_size

    # Delayed computation of the activations of a batch
    @dask.delayed
    def batch_activation():
        batch_images, _ = next(batch_gen())
        return activation_function([batch_images, 0])[0]

    # Delayed iteration over the data set
    activations_delayed = [batch_activation() for _
            in range(batch_gen.n_batches)]
    activations_da_list = [da.from_delayed(
            activation_delayed,
            shape=(batch_size * n_aug, ) + layer_shape,
            dtype=K.floatx())
        for activation_delayed in activations_delayed]
    activations_da = da.concatenate(activations_da_list, axis=0)

    # The last batch can be smaller
    activations_da = activations_da[:n_images * n_aug]

    # Reshape the activations such that 
    # shape = (n_diff_images, layer_dim, n_aug)
    activations_da = da.reshape(activations_da, 
                                (activations_da.shape[0], layer_dim))
    activations_da = da.transpose(da.reshape(activations_da.T, 
                                             (layer_dim, n_images, n_aug)),
                                  (1, 0, 2))

    return activations_da
