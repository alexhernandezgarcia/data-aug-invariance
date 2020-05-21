"""
Routine for performing "DreamLearning": post-training a pre-trained neural
network with in a layer-wise fashion by inserting noise layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import keras.backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
from keras.metrics import top_k_categorical_accuracy
from keras.layers import Input
from keras.layers import Flatten, BatchNormalization
from keras.layers import add, subtract, multiply, dot
from keras.layers import Lambda
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.losses import mean_squared_error

import keras.losses

from functools import partial, update_wrapper

import tensorflow as tf

import h5py
import yaml

import sys
import os
import re

import argparse
import time
import shutil
from tqdm import tqdm

# Initialize the Flags container
FLAGS = None


def main(argv=None):

    # Set test phase
    K.set_learning_phase(0)

    # Set float default
    K.set_floatx('float32')

    handle_train_dir(FLAGS.train_dir)
    
    _print_header()

    surgery()


def surgery():

    # Output file
    log_file = open(os.path.join(FLAGS.train_dir, 'log'), 'w')

    # Open training configuration file
    with open(FLAGS.train_config_file, 'r') as yml_file:
        train_config = yaml.load(yml_file, Loader=yaml.FullLoader)
    batch_size = train_config['batch_size']

    # Open HDF5 file containing the data set and get images and labels
    hdf5_file = h5py.File(FLAGS.data_file, 'r')
    if (FLAGS.seed is not None) & (FLAGS.pct_test != 1.0):
        shuffle = True
    else: 
        shuffle = False
    images, labels, hdf5_aux = data_input.hdf52dask(hdf5_file, FLAGS.group, 
                                               FLAGS.chunk_size, shuffle, 
                                               FLAGS.seed, FLAGS.pct_test)

    # Image parameters
    with open(FLAGS.image_params_file, 'r') as yml_file:
        train_image_params = yaml.load(yml_file, Loader=yaml.FullLoader)
        if train_image_params['do_random_crop'] & \
           (train_image_params['crop_size'] is not None):
            image_shape = train_image_params['crop_size']
    val_image_params = data_input.validation_image_params(**train_image_params)

    # Attack parameters
    with open(FLAGS.attack_params_file, 'r') as yml_file:
        attack_params = yaml.load(yml_file, Loader=yaml.FullLoader)

    # Load original model 
    model = load_model(os.path.join(FLAGS.model))
    model = ensure_softmax_output(model)
    model.summary()
    model.summary(print_fn=lambda x: log_file.write(x + '\n'))

    # Load adversarial model
    if FLAGS.model_adv:
        model_adv = load_model(os.path.join(FLAGS.model_adv))
    else:
        model_adv = model

    # Compute original clean accuracy
    if FLAGS.test_orig:
        compute_accuracy(model, images, labels, batch_size,
                         val_image_params, None, log_file, orig_new='orig')
    
    # Compute original adversarial accuracy
    if FLAGS.test_adv_orig:
        # White-box
        compute_adv_accuracy(model, model, images, labels, batch_size,
                             val_image_params, attack_params, log_file, 
                             orig_new='orig')
        model = del_extra_nodes(model, verbose=0)
        # Black-box
        if FLAGS.model_adv:
            compute_adv_accuracy(model, model_adv, images, labels, batch_size,
                                 val_image_params, attack_params, log_file, 
                                 orig_new='orig')

    
    # Create new model by modifying the logits
    model = del_extra_nodes(model, verbose=0)
    print('\nCreating new model...')
    if 'bn' in train_config['key_layer']:
        new_model = insert_bn(model, train_config['key_layer'], 
                              n_bn=train_config['n_layers'])
    else:
        new_model = insert_layer_old(model, train_config['key_layer'])
    # new_model.compile(loss='mean_squared_error', optimizer='sgd')

    # Print summary architecture
    if FLAGS.print_summary:
        new_model.summary()
    new_model.summary(print_fn=lambda x: log_file.write(x + '\n'))

    # Save new model
    if FLAGS.save_new:
        model_filename = os.path.join(FLAGS.train_dir,
                                      'model_new_' +
                                      time.strftime('%a_%d_%b_%Y_%H%M%S'))
        new_model.save(model_filename)

    # Compute new clean accuracy
    if FLAGS.test_new:
        compute_accuracy(new_model, images, labels, batch_size,
                         val_image_params, None, log_file, orig_new='new')
    
    # Compute new adversarial accuracy
    if FLAGS.test_adv_new:
        # White-box
        compute_adv_accuracy(new_model, new_model, images, labels, batch_size,
                             val_image_params, attack_params, log_file, 
                             orig_new='new')
        new_model = del_extra_nodes(new_model, verbose=0)
        # Black-box
        if FLAGS.model_adv:
            compute_adv_accuracy(new_model, model_adv, images, labels, 
                                 batch_size, val_image_params, attack_params, 
                                 log_file, orig_new='new')

    # Close HDF5 File and log file
    hdf5_file.close()
    log_file.close()

    # Close and remove aux HDF5 files
    for f in hdf5_aux:
        filename = f.filename
        f.close()
        os.remove(filename)


def categorical_crossentropy_from_logits(y_true, logits):
    return K.categorical_crossentropy(y_true, logits, from_logits=True)


def del_extra_nodes(model, log_file=None, verbose=0):
    """
    Remove all extra nodes (i.e. all but the first nodes) from all layers of 
    the network to avoid undesirable behaviour.

    Parameters
    ----------
    model : Keras Model
        The original model

    verbose : int
        Verbosity. It prints and writes log messages if larger than 0.

    Returns
    -------
    model : Keras Model
        The modified model
    """
    if model.inbound_nodes:
        if len(model.inbound_nodes) > 1:
            model.inbound_nodes = [model.inbound_nodes[0]]
            if verbose > 0:
                print('\nRemoved inbound nodes at model input')
            if log_file:
                log_file.write('\nRemoved inbound nodes at model input')
    if model.outbound_nodes:
        if len(model.outbound_nodes) > 1:
            model.outbound_nodes = [model.outbound_nodes[0]]
            if verbose > 0:
                print('\nRemoved outbound nodes at model output')
            if log_file:
                log_file.write('\nRemoved outbound nodes at model output')
    for layer in model.layers:
        if layer.inbound_nodes:
            if len(layer.inbound_nodes) > 1:
                layer.inbound_nodes = [layer.inbound_nodes[0]]
                if verbose > 0:
                    print('\nRemoved inbound nodes at layer %s' % layer.name)
            if log_file:
                    log_file.write('\nRemoved inbound nodes at layer '
                                   '%s' % layer.name)
        if layer.outbound_nodes:
            if len(layer.outbound_nodes) > 1:
                layer.outbound_nodes = [layer.outbound_nodes[0]]
                if verbose > 0:
                    print('\nRemoved outbound nodes at layer %s' % layer.name)
            if log_file:
                    log_file.write('\nRemoved outbound nodes at layer '
                                   '%s' % layer.name)

    return model


def network2dict(model):
    """
    Stores the inbound and outbound nodes of each layer into a dictionary.

    Parameters
    ----------
    model : Keras Model
        The original model

    Returns
    -------
    network_dict : dict
        A dictionary specifying the input layers of each tensor and the output 
        tensors of each layer
    """

    network_dict = {'model': 
            {'inbound_nodes': [id(node) for node in model.inbound_nodes],
             'outbound_nodes': [id(node) for node in model.outbound_nodes]}}

    for layer in model.layers:
        network_dict.update({layer.name: 
	    {'inbound_nodes': [id(node) for node in layer.inbound_nodes],
            'outbound_nodes': [id(node) for node in layer.outbound_nodes]}})

    return network_dict


def restore_nodes(model, network_dict):
    """
    Restores the inbound and outbound nodes of each layer according to a 
    reference dictionary

    Parameters
    ----------
    model : Keras Model
        The original model
        
    network_dict : dict
        A dictionary containing the nodes to be restored

    Return
    ------
    model : Keras Model
        The model with restored nodes
    """
    # Model inbound nodes
    nodes = [id(node) for node in model.inbound_nodes]
    common_nodes = set(nodes).intersection(
            set(network_dict['model']['inbound_nodes']))
    old_nodes = set(nodes).difference(
            set(network_dict['model']['inbound_nodes']))
    if len(old_nodes) > 0:
        for node in model.inbound_nodes:
            if id(node) in old_nodes:
                model.inbound_nodes.remove(node)
    if len(common_nodes) > 0:
        model.inbound_nodes = [node for node in model.inbound_nodes 
                if id(node) in common_nodes]
        
    # Model outbound nodes
    nodes = [id(node) for node in model.outbound_nodes]
    common_nodes = set(nodes).intersection(
            set(network_dict['model']['outbound_nodes']))
    old_nodes = set(nodes).difference(
            set(network_dict['model']['outbound_nodes']))
    if len(old_nodes) > 0:
        for node in model.outbound_nodes:
            if id(node) in old_nodes:
                model.outbound_nodes.remove(node)
    if len(common_nodes) > 0:
        model.outbound_nodes = [node for node in model.outbound_nodes 
                if id(node) in common_nodes]
        
    # Iterate over the layers
    for layer in model.layers:
        # Inbound nodes
        nodes = [id(node) for node in layer.inbound_nodes]
        common_nodes = set(nodes).intersection(
                set(network_dict[layer.name]['inbound_nodes']))
        old_nodes = set(nodes).difference(
                set(network_dict[layer.name]['inbound_nodes']))
        if len(old_nodes) > 0:
            for node in layer.inbound_nodes:
                if id(node) in old_nodes:
                    layer.inbound_nodes.remove(node)
        if len(nodes) == 0:
            pass
        elif len(common_nodes) > 0:
            layer.inbound_nodes = [node for node in layer.inbound_nodes 
                    if id(node) in common_nodes]
        else:
            raise ValueError('No common inbound nodes between the dictionary '
                             'and the current graph at layer {}'.format(
                                 layer.name))

        # Outbound nodes
        nodes = [id(node) for node in layer.outbound_nodes]
        common_nodes = set(nodes).intersection(
                set(network_dict[layer.name]['outbound_nodes']))
        old_nodes = set(nodes).difference(
                set(network_dict[layer.name]['outbound_nodes']))
        if len(old_nodes) > 0:
            for node in layer.outbound_nodes:
                if id(node) in old_nodes:
                    layer.outbound_nodes.remove(node)
        if len(nodes) == 0:
            pass
        elif len(common_nodes) > 0:
            layer.outbound_nodes = [node for node in layer.outbound_nodes 
                    if id(node) in common_nodes]
        else:
            raise ValueError('No common outbound nodes between the dictionary '
                             'and the current graph at layer {}'.format(
                                 layer.name))

    return model


def del_old_nodes(model, network_dict):
    """
    Updates the inbound and outbound nodes of each layer by keeping only the 
    nodes that are not present in the older reference dictionary

    Parameters
    ----------
    model : Keras Model
        The original model
        
    network_dict : dict
        An older reference network dictionary

    Return
    ------
    model : Keras Model
        The model with updated nodes
    """
    # Model inbound nodes
    nodes = [id(node) for node in model.inbound_nodes]
    diff_nodes = set(nodes).difference(
            set(network_dict['model']['inbound_nodes']))
    if len(diff_nodes) > 0:
        model.inbound_nodes = [node for node in model.inbound_nodes 
                if id(node) in diff_nodes]
        
    # Model outbound nodes
    nodes = [id(node) for node in model.outbound_nodes]
    diff_nodes = set(nodes).difference(
            set(network_dict['model']['outbound_nodes']))
    if len(diff_nodes) > 0:
        model.outbound_nodes = [node for node in model.outbound_nodes 
                if id(node) in diff_nodes]
        
    # Iterate over the layers
    for layer in model.layers:
        if layer.name in network_dict:
            # Inbound nodes
            nodes = [id(node) for node in layer.inbound_nodes]
            diff_nodes = set(nodes).difference(
                    set(network_dict[layer.name]['inbound_nodes']))
            if len(diff_nodes) > 0:
                layer.inbound_nodes = [node for node in layer.inbound_nodes 
                        if id(node) in diff_nodes]

            # Outbound nodes
            nodes = [id(node) for node in layer.outbound_nodes]
            diff_nodes = set(nodes).difference(
                    set(network_dict[layer.name]['outbound_nodes']))
            if len(diff_nodes) > 0:
                layer.outbound_nodes = [node for node in layer.outbound_nodes 
                        if id(node) in diff_nodes]

    _update_keras_history(model)
                
    return model


def _update_keras_history(model):
    def _update_node_history(nodes):
        for idx_node, node in enumerate(nodes):
            if len(node.node_indices) > 0:
                node.node_indices[idx_node] = idx_node
            for idx_tensor, tensor in enumerate(node.input_tensors):
                tensor._keras_history = (tensor._keras_history[0], idx_node,
                                         idx_tensor)
            for idx_tensor, tensor in enumerate(node.output_tensors):
                tensor._keras_history = (tensor._keras_history[0], idx_node,
                                         idx_tensor)

    _update_node_history(model.inbound_nodes)
    _update_node_history(model.outbound_nodes)

    for layer in model.layers:
        _update_node_history(layer.inbound_nodes)
        _update_node_history(layer.outbound_nodes)
        

def del_mse_nodes(model, log_file=None, verbose=1):
    """
    Remove all the node branches corresponding to the pairwise MSE layers so as
    to keep the object recognition graph only.

    Parameters
    ----------
    model : Keras Model
        The original model

    verbose : int
        Verbosity. It prints and writes log messages if larger than 0.

    Returns
    -------
    model : Keras Model
        The modified model
    """
    def del_nonrelevant_nodes(model):
        relevant_nodes = []
        for node in model.nodes_by_depth.values():
            relevant_nodes.extend(node)

        for layer in model.layers:
            if layer.inbound_nodes:
                inbound_nodes = layer.inbound_nodes
                for node in inbound_nodes:
                    if node not in relevant_nodes:
                        inbound_nodes.remove(node)
                        if verbose > 0:
                            print('Removed (mse) node {} from layer {}'.format(
                                node, layer.name))
            if layer.outbound_nodes:
                outbound_nodes = layer.outbound_nodes
                for node in outbound_nodes:
                    if node not in relevant_nodes:
                        outbound_nodes.remove(node)
                        if verbose > 0:
                            print('Removed (mse) node {} from layer {}'.format(
                                node, layer.name))

    def del_extra_nodes(layer, softmax):
        if layer.outbound_nodes:
            outbound_nodes = layer.outbound_nodes
            for outnode in outbound_nodes:
                if outnode.outbound_layer:
                    if not connected_to_softmax(
                            outnode.outbound_layer, softmax):
                        outbound_nodes.remove(outnode)
                        if verbose > 0:
                            print('Removed (mse) node {} from layer {}'.format(
                                outnode, layer.name))
                elif outnode.outbound_layers:
                    for out_layer in outnode.outbound_layers:
                        if not connected_to_softmax(out_layer, softmax):
                            outbound_nodes.remove(outnode)
                        if verbose > 0:
                            print('Removed (mse) node {} from layer {}'.format(
                                innode, layer.name))
                else:
                    pass
        if layer.inbound_nodes:
            inbound_nodes = layer.inbound_nodes
            for innode in inbound_nodes:
                if innode.inbound_layers:
                    for in_layer in innode.inbound_layers:
                        del_extra_nodes(in_layer, softmax)
                else:
                    pass
        return 

    def connected_to_softmax(layer, softmax):
        if layer is softmax:
            return True
        else:
            if layer.outbound_nodes:
                outbound_nodes = layer.outbound_nodes
                for outnode in outbound_nodes:
                    if outnode.outbound_layer:
                        return connected_to_softmax(outnode.outbound_layer, 
                                                    softmax)
                    elif outnode.outbound_layers:
                        for out_layer in outnode.outbound_layers:
                            return connected_to_softmax(out_layer, softmax)
                    else:
                        False
            else:
                return False
            
    def connected_to_input(layer, input):
        if layer is input:
            return True
        else:
            if layer.inbound_nodes:
                inbound_nodes = layer.inbound_nodes
                for innode in inbound_nodes:
                    if innode.inbound_layers:
                        for in_layer in innode.inbound_layers:
                            return connected_to_input(in_layer, input)
                    else:
                        False
            else:
                return False
            
    # Delete all outputs except Softmax
    if len(model.outputs) > 1:
        for layer, name, tensor, node_index, tensor_index in zip(
                    model.output_layers,
                    model.output_names,
                    model.outputs,
                    model.output_layers_node_indices,
                    model.output_layers_tensor_indices):
            if layer.name == 'softmax':
                model.output_layers = [layer]
                model.output_names = [name]
                model.outputs = [tensor]
                model.output_layers_node_indices = [node_index]
                model.output_layers_tensor_indices = [tensor_index]
                break
            else:
                pass

        # Delete node from categorical model
        del_nonrelevant_nodes(model)

        del_extra_nodes(model.output_layers[0], model.output_layers[0])
        new_model = Model(inputs=model.input, outputs=model.outputs[0])
    else:
        new_model = model

    return new_model


def ensure_softmax_output(model):
    """
    Adds a softmax layer on top of the logits layer, in case the output layer 
    is a logits layer.

    Parameters
    ----------
    model : Keras Model
        The original model

    Returns
    -------
    new_model : Keras Model
        The modified model
    """
    
    if 'softmax' not in model.output_names:
        if 'logits' in model.output_names:
            output = Activation('softmax', name='softmax')(model.output)
            new_model = Model(inputs=model.input, outputs=output)
        else:
            raise ValueError('The output layer is neither softmax nor logits')
    else:
        new_model = model


    return new_model


def insert_layer(model, layer_regex, insert_layer_factory,
                 insert_layer_name=None, position='after'):
    """
    Inserts a layer before, after or replacing the layer(s) specified in the 
    arguments through a regular expression.

    Parameters
    ----------
    model : Keras Model
        The original model

    layer_regex : str
        Name of the key layer, as a regular expression.

    insert_layer_factory : func
        Factory for the layers to be inserted

    insert_layer_name : str
        If None, the new layers' names will be the concatenation of the key
        layer's name and the original insert_layer name. Otherwise, 
        insert_layer_name is used as name.

    position : str
        Specifies whether the new layers are inserted before, after or
        replacing the key layers.

    Returns
    -------
    Keras Model
        The modified new model
    """

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)


def network2dict_old(model):
    """
    Parses the architecture of a network into a dictionary, .

    Parameters
    ----------
    model : Keras Model
        The original model

    Returns
    -------
    network_dict : dict
        A dictionary specifying the input layers of each tensor and the output 
        tensors of each layer
    """

    network_dict = {'input_layers_of': {}, 'output_tensor_of': {}}

    for idx, layer in enumerate(model.layers):
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

        if idx == 0:
            tensor = model.input
            network_dict['output_tensor_of'].update(
                    {layer.name: tensor})
        else:
            tensor = layer(tensor)
            network_dict['output_tensor_of'].update({layer.name: tensor})

    return network_dict



def insert_layer_old(model, key_layer_name):
    """
    Insert a new layer right after the layer specified in the argument.

    Parameters
    ----------
    model : Keras Model
        The original model

    key_layer_name : str
        Name of the layer after which the new one is inserted.

    Returns
    -------
    new_model : Keras Model
        The modified new model
    """
    
    # Determine output layer of the model
    output_layer = model.get_layer('softmax')

    # Extract information from the key key layer, after which the new layer
    # will be inserted
    key_layer = model.get_layer(key_layer_name)
    x = key_layer.output

    # Determine next layer after the key layer
    next_layer = key_layer.outbound_nodes[0].outbound_layer
    key_layer.outbound_nodes = []

    # Attach the new layers
    n_bn = 1
    for n in range(n_bn):
        alpha = 1e9
#             beta = 1.
#         beta = 1000.
        beta = 0.
#         x = Lambda(lambda x: alpha * x + beta)(x)
#         x = Lambda(lambda x: alpha * (x - K.min(x)) / (K.max(x) - K.min(x)))(x)
#         x = Lambda(lambda x: K.max(x))(x)
        temp = 0.001
        x = Lambda(lambda x: x / temp)(x)
#         x = Lambda(lambda x: alpha * K.batch_normalization(x, mean=K.mean(x, axis=0), var=K.var(x, axis=0), beta=0., gamma=1.))(x)
#             x = multiply([x, x], name='multiply{}'.format(n+1))
#             x = add([x, x], name='add{}'.format(n+1))

    # Connect the new layers to the original 'next_layer'
    next_layer.inbound_nodes = []
    x = next_layer(x)
    
    # Re-connect the rest of the layers
    while next_layer is not output_layer:
        next_layer = next_layer.outbound_nodes[0].outbound_layer
        next_layer.inbound_nodes[0].inbound_layers[0].outbound_nodes = []
        next_layer.inbound_nodes = []
        x = next_layer(x)
        
    new_model = Model(inputs=model.input, outputs=x)

    return new_model


def insert_bn(model, bn_layer_str, n_bn):
    """
    Insert a chain of Batch Normalization layers right after the existing Batch
    Normalization layer specified in the argument.

    Parameters
    ----------
    model : Keras Model
        The original model

    bn_layer_str : str
        Name of the batch normalization layer after which the new one is
        inserted.

    n_bn : int
        Number of new BatchNorm layers to be inserted

    Returns
    -------
    new_model : Keras Model
        The modified new model
    """
    
    # Determine output layer of the model
    output_layer = model.get_layer('softmax')

    # Extract information from the key BN layer, after which the new BN layers
    # will be inserted
    bn_layer = model.get_layer(bn_layer_str)
    x = bn_layer.output
    w_bn = bn_layer.get_weights()
    gamma = w_bn[0]
    beta = w_bn[1]

    # Determine next layer after the key BN layer
    next_layer = bn_layer.outbound_nodes[0].outbound_layer
    bn_layer.outbound_nodes = []

    # Attach the new BatchNorm layers
    for n in range(n_bn):
        x = BatchNormalization(name= '{}_new{}'.format(bn_layer_str, n+1))(x)

    # Connect the new layers to the original 'next_layer'
    next_layer.inbound_nodes = []
    x = next_layer(x)
    
    # Re-connect the rest of the layers
    while next_layer is not output_layer:
        next_layer = next_layer.outbound_nodes[0].outbound_layer
        next_layer.inbound_nodes[0].inbound_layers[0].outbound_nodes = []
        next_layer.inbound_nodes = []
        x = next_layer(x)
        
    new_model = Model(inputs=model.input, outputs=x)

    # Set weights of the new batch normalization layers
    for n in range(n_bn):
        new_bn_layer = new_model.get_layer(bn_layer_str + '_new{}'.format(n+1))
        new_bn_layer.set_weights(w_bn)

    return new_model


def insert_layer_old(model, new_layer, prev_layer_name):
    
    x = model.input
    for layer in model.layers:
        x = layer(x)
        if layer.name == prev_layer_name:
            x = new_layer(x)

    new_model = Model(inputs=model.input, outputs=x)


def ablate_activations(model, layer_regex, pct_ablation, seed):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        x = layer(layer_input)

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            ablation_layer = Dropout(
                rate=pct_ablation, 
                noise_shape=(None, ) + tuple(np.repeat(1,
                    len(layer.output_shape) - 2)) + (layer.output_shape[-1], ),
                seed=seed, name='{}_dropout'.format(layer.name))
            x = ablation_layer(x, training=True)
            if seed:
                seed += 1

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)


def compute_accuracy(model, images, labels, batch_size, image_params, 
                     attack_params, log_file, orig_new):
    """
    Computes the accuracy of a model, either on the clean data set.

    Parameters
    ----------
    model : Keras Model
        The model on which the accuracy is computed

    images : dask array
        The set of images

    labels : dask array
        The ground truth labels

    batch_size : int
        Batch size

    image_params: dict
        Dictionary of data augmentation parameters

    attack_params: dict
        Dictionary of the attack parameters

    log_file : File
        The log file 

    orig_new : str
        Either 'orig' or 'new'. Only used for print and log output.
    """
    if orig_new == 'orig':
        orig_new_str = 'original'
    elif orig_new == 'new':
        orig_new_str = 'new'
    else:
        raise NotImplementedError

    print('\nComputing {} clean accuracy...'.format(orig_new_str))

    # Compute accuracy
    results_dict = test.test(images, labels, batch_size, model, image_params, 
                             repetitions=1)
    acc = results_dict['mean_acc'].compute()

    # Print
    print('{} clean accuracy: {:.4f}'.format(orig_new_str.title(), acc))

    # Write to log file
    log_file.write('{} clean accuracy: '
                   '{:.4f}\n'.format(orig_new_str.title(), acc))

    
def compute_adv_accuracy(model, model_adv, images, labels, batch_size, 
                         image_params, attack_params, log_file, orig_new):
    """
    Computes the accuracy of a model, either on the clean data set (if 
    adversarial is False), or on adversarial examples (if adversarial is True).

    Parameters
    ----------
    model : Keras Model
        The model on which the accuracy is computed

    model_adv : Keras Model
        The model used to generate adversarial examples

    images : dask array
        The set of images

    labels : dask array
        The ground truth labels

    batch_size : int
        Batch size

    image_params: dict
        Dictionary of data augmentation parameters

    attack_params: dict
        Dictionary of the attack parameters

    log_file : File
        The log file 

    orig_new : str
        Either 'orig' or 'new'. Only used for print and log output.
    """
    if orig_new == 'orig':
        orig_new_str = 'original'
    elif orig_new == 'new':
        orig_new_str = 'new'
    else:
        raise NotImplementedError

    if model == model_adv:
        attack_type = 'white-box'
    else:
        attack_type = 'black-box'

    print('\nComputing {} {} adversarial accuracy...'.format(orig_new_str,
                                                             attack_type))

    # Compute accuracy and MSE
    adv_acc, adv_mse = test_adv.test(images, labels, batch_size,
                                     model, model_adv, image_params, 
                                     attack_params, do_print=False)

    # Print
    print('{} {} adversarial accuracy: {:.4f}'.format(orig_new_str.title(), 
                                                      attack_type, adv_acc))
    print('{} MSE between adversarial and clean images: '
          '{:.4f}\n'.format(orig_new_str.title(), adv_mse))

    # Write to log file
    log_file.write('{} {} adversarial accuracy: '
                   '{:.4f}\n'.format(orig_new_str.title(), attack_type, 
                                     adv_acc))
    log_file.write('{} MSE between adversarial and clean images: '
                   '{:.4f}\n\n'.format(orig_new_str.title(), adv_mse))



def _print_header(write_file=True):
    """
    Prints a summary of the current execution and writes it to a txt file
    """
    print('')
    print('--------------------------------------------------')
    print('Running %s' % sys.argv[0])
    print('')
    print('FLAGS:')
    flags = vars(FLAGS)
    for k in flags.keys():
        print('%s: %s' % (k, flags[k]))

    # Print image pre-processing parameters
    print('')
    print('Image pre-processing parameters:')
    with open(flags['image_params_file'], 'r') as yml_file:
        daug_params = yaml.load(yml_file, Loader=yaml.FullLoader)
    with open(flags['train_config_file'], 'r') as yml_file:
        train_params = yaml.load(yml_file, Loader=yaml.FullLoader)
    for param in train_params:
        print('\t%s: %s' % (param, train_params[param]))
    print('')
    for param in daug_params:
        print('\t%s: %s' % (param, daug_params[param]))
    print('')
    print('--------------------------------------------------')

    if write_file:
        output_file = os.path.join(FLAGS.train_dir, 'flags_' +
                                   time.strftime('%a_%d_%b_%Y_%H%M%S'))
        with open(output_file, 'w') as f:
            f.write('FLAGS:\n')
            flags = vars(FLAGS)
            for k in flags.keys():
                f.write('%s: %s\n' % (k, flags[k]))
            f.write('Training parameters:\n')
            for param in train_params:
                f.write('\t%s: %s\n' % (param, train_params[param]))
            f.write('Image pre-processing parameters:\n')
            for param in daug_params:
                f.write('\t%s: %s\n' % (param, daug_params[param]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_file',
        type=str,
        default='/mnt/data/alex/datasets/hdf5/cifar10.hdf5',
        help='Path to the HDF5 file containing the data set.'
    )
    parser.add_argument(
        '--group',
        type=str,
        default='cifar10_test',
        help='Group name in the HDF5 file indicating the test data set.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/dreamlearning/',
        help='Directory where to write event logs and checkpoint'
    )
    parser.add_argument(
        '--pct_val',
        type=float,
        default=0.2,
        help='Percentage of samples for the validation set'
    )
    parser.add_argument(
        '--pct_train',
        type=float,
        default=1.0,
        help='Percentage of examples to use from the training set.'
    )
    parser.add_argument(
        '--metrics',
        type=str,  # list of str
        nargs='*',  # 0 or more arguments can be given
        default=None,
        help='List of metrics to monitor, separated by spaces'
    )
    parser.add_argument(
        '--train_config_file',
        type=str,
        default="config.yml",
        help='Path to the training configuration file'
    )
    parser.add_argument(
        '--image_params_file',
        type=str,
        default="base_image_config.yml",
        help='Path to the configuration file with the image pre-processing'
             'parameters'
    )
    parser.add_argument(
        '--attack_params_file',
        type=str,
        default="attacks/fgsm_eps03.yml",
        help='Path to the configuration file with the attack parameters'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model file (architecture + weights + optimizer state)'
    )
    parser.add_argument(
        '--model_adv',
        type=str,
        default=None,
        help='Model file (architecture + weights + optimizer state) used to '
             'generate the adversarial examples'
    )
    parser.add_argument(
        '--test_new',
        action='store_true',
        dest='test_new',
        help='Compute test accuracy of new model'
    )
    parser.add_argument(
        '--notest_new',
        action='store_false',
        dest='test_new',
        help='Do not compute test accuracy of new model'
    )
    parser.set_defaults(test_new=True)
    parser.add_argument(
        '--test_adv_new',
        action='store_true',
        dest='test_adv_new',
        help='Compute test adversarial accuracy of new model'
    )
    parser.add_argument(
        '--notest_adv_new',
        action='store_false',
        dest='test_adv_new',
        help='Do not compute test adversarial accuracy of new model'
    )
    parser.set_defaults(test_adv_new=True)
    parser.add_argument(
        '--test_orig',
        action='store_true',
        dest='test_orig',
        help='Compute test accuracy of original model'
    )
    parser.add_argument(
        '--notest_orig',
        action='store_false',
        dest='test_orig',
        help='Do not compute test accuracy of original model'
    )
    parser.set_defaults(test_orig=False)
    parser.add_argument(
        '--test_adv_orig',
        action='store_true',
        dest='test_adv_orig',
        help='Compute test adversarial accuracy of original model'
    )
    parser.add_argument(
        '--notest_adv_orig',
        action='store_false',
        dest='test_adv_orig',
        help='Do not compute test adversarial accuracy of original model'
    )
    parser.set_defaults(test_adv_orig=False)
    parser.add_argument(
        '--save_new',
        action='store_true',
        dest='save_new',
        help='Save the model (architecture + weights + optimizer state) at the'
             'end of each dreamt layer.'
    )
    parser.add_argument(
        '--print_summary',
        action='store_true',
        dest='print_summary',
        help='Print the model summary architecture'
    )
    parser.set_defaults(print_summary=False)
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help='Size of the dask array chunks'
    )
    parser.add_argument(
        '--pct_test',
        type=float,
        default=1.0,
        help='Percentage of examples to use from the test set.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for the test set shuffling'
    )
    parser.add_argument(
        '--log_device_placement',
        action='store_true',
        dest='log_device_placement',
        help='Enable log device placement.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
