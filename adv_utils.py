from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow.compat.v1.keras.backend as K
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.models import load_model

import numpy as np
import h5py
import dask.array as da
import yaml
import sys
import os
import argparse
from tqdm import tqdm

import data_input
from utils import pairwise_loss
## Import the whole compat version of keras to set the losses =================
import tensorflow.compat.v1.keras as keras
## ============================================================================
keras.losses.pairwise_loss = pairwise_loss
keras.losses.custom_mse = pairwise_loss
from surgery import del_mse_nodes, ensure_softmax_output

from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod, DeepFool
from cleverhans.attacks import CarliniWagnerL2, SPSA
try:
    from cleverhans.attacks import ProjectedGradientDescent
except:
    from cleverhans.attacks import MadryEtAl as ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper

# Initialize the Flags container
FLAGS = None


def main(argv=None):

    # Set test phase
    K.set_learning_phase(0)

    # Set float default
    K.set_floatx('float32')

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    K.set_session(sess)
    
    _print_flags()

    # Define output file
    if FLAGS.do_write:
        output_file = os.path.join(os.path.dirname(FLAGS.model),
                                   'advacc_' +
                                   os.path.basename(FLAGS.model) + '_' +
                                   os.path.basename(FLAGS.attack_params_file)\
                                                    .split('.')[0])
    else:
        output_file = None
    
    # Load model
    model = load_model(os.path.join(FLAGS.model))
    model = del_mse_nodes(model, verbose=1)
    model = ensure_softmax_output(model)
    
    # Load adversarial model
    if FLAGS.model_adv:
        model_adv = load_model(os.path.join(FLAGS.model_adv))
    else:
        model_adv = model

    # Open HDF5 file containing the data set and get images and labels
    hdf5_file = h5py.File(FLAGS.data_file, 'r')
    if (FLAGS.seed is not None) & (FLAGS.pct_test != 1.0):
        shuffle = True
    else: 
        shuffle = False
    images, labels, hdf5_aux = data_input.hdf52dask(hdf5_file, FLAGS.group, 
                                               FLAGS.chunk_size, shuffle, 
                                               FLAGS.seed, FLAGS.pct_test)

    # Load image parameters
    with open(FLAGS.image_params_file, 'r') as yml_file:
        train_image_params = yaml.load(yml_file, Loader=yaml.FullLoader)
    image_params_dict = data_input.validation_image_params(
            **train_image_params)

    # Load attack parameters
    with open(FLAGS.attack_params_file, 'r') as yml_file:
        attack_params_dict = yaml.load(yml_file, Loader=yaml.FullLoader)

    test_rep_orig(FLAGS.data_file, FLAGS.group, FLAGS.chunk_size,
                  FLAGS.batch_size, model, train_image_params, 
                  train_image_params, 1, None, [])

    test(images, labels, FLAGS.batch_size, model, model_adv, image_params_dict, 
         attack_params_dict, output_file)

    # Close HDF5 File
    hdf5_file.close()

    # Close and remove aux HDF5 files
    for f in hdf5_aux:
        filename = f.filename
        f.close()
        os.remove(filename)


def test(images, labels, batch_size, model, model_adv, image_params_dict, 
         attack_params_dict, output_file=None, do_print=True):
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

    model_adv : Keras Model
        The model used to generate adversarial examples

    image_params_dict : dict
        Dictionary of data augmentation parameters

    attack_params_dict : dict
        Dictionary of the attack parameters

    output_file : str or None
        The outfile to write the results 

    do_print : bool
        Whether to print the adversarial accuracy and MSE

    Returns
    -------
    results_dict : dict
        Dictionary containing some performance metrics
    """

    # Set test phase and get session
    sess = K.get_session()
    if isinstance(K.learning_phase(), int):
        learning_phase = K.learning_phase()
        K.set_learning_phase(0)
    
    # Initialize adversarial attack
    attack, attack_params, bs = init_attack(model_adv, attack_params_dict, 
                                            sess)
    if bs:
        batch_size = bs

    # Create batch generator
    image_gen = data_input.get_generator(images, **image_params_dict)
    batch_gen = batch_generator(image_gen, images, labels, batch_size, 
                                aug_per_im=1, shuffle=False)
    n_batches_per_epoch = int(np.ceil(float(images.shape[0]) / batch_size))

    # Define input TF placeholder
    if image_params_dict['crop_size']:
        image_shape = image_params_dict['crop_size']
    else:
        image_shape = images.shape[1:]
    x = tf.placeholder(K.floatx(), shape=(bs,) + tuple(image_shape))
    y = tf.placeholder(K.floatx(), shape=(bs,) + (labels.shape[-1],))

    # Define adversarial predictions symbolically
    x_adv = attack.generate(x, **attack_params)
    x_adv = tf.stop_gradient(x_adv)
    predictions_adv = model(x_adv)

    # Define accuracy symbolically
    correct_preds = tf.equal(tf.argmax(y, axis=-1), 
                             tf.argmax(predictions_adv, axis=-1))
    acc_value = tf.reduce_mean(tf.to_float(correct_preds))

    # Define mean squared error symbolically
    mse_value = tf.reduce_mean(tf.square(tf.subtract(x, x_adv)))

    # Init results variables
    accuracy = 0.0
    mse = 0.0

    # Initialize matrix to store the predictions.
#     predictions = np.zeros([images.shape[0], labels.shape[-1]])

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
            batch_acc = acc_value.eval(feed_dict={x: batch[0], y: yy})

            # Evaluate MSE
            batch_mse = mse_value.eval(feed_dict={x: batch[0]})

            # Adversarial predictions
#             predictions[init:init+this_batch_size, :] = \
#                     predictions_adv.eval(feed_dict={x: batch[0]})

            # Update accuracy and MSE
            accuracy += (this_batch_size * batch_acc)
            mse += (this_batch_size * batch_mse)

            init += this_batch_size

    accuracy /= images.shape[0]
    mse /= images.shape[0]

    # Compute accuracy
#     acc_post = np.divide(np.sum(np.argmax(predictions, axis=1) == \
#                                 np.argmax(labels, axis=1)), 
#                          float(images.shape[0]))


    if do_print:
        print('Aversarial accuracy against %s: %.4f' %
              (attack_params_dict['attack'], accuracy))
#         print('Aversarial accuracy against %s: %.4f' %
#               (attack_params_dict['attack'], acc_post))
        print('MSE between %s adversaries and originals: %.4f\n' %
              (attack_params_dict['attack'], mse))

    if output_file is not None:
        _write_results(accuracy, output_file)

    # Set original phase
    if isinstance(K.learning_phase(), int):
        K.set_learning_phase(learning_phase)

    return accuracy, mse


def init_attack(model, attack_params_dict, sess):
    """
    Initialize the adversarial attack using the cleverhans toolbox

    Parameters
    ----------
    model : Keras Model
        The model to attack

    attack_params_dict : dict
        Self-defined dictionary specifying the attack and its parameters

    sess : Session
        The current tf session

    Returns
    -------
    attack : cleverhans Attack
        The Attack object

    attack_params
        Dictionary with the value of the attack parameters, valid to generate
        adversarial examples with cleverhans.
    """

    # Wrapper for the Keras model
    model_wrap = KerasModelWrapper(model)

    # Initialize attack
    batch_size = None
    if attack_params_dict['attack'] == 'fgsm':
        attack = FastGradientMethod(model_wrap, sess=sess)
        attack_params = {'eps': attack_params_dict['eps'], 'clip_min': 0., 
                         'clip_max': 1.}
    elif attack_params_dict['attack'] == 'spsa':
        attack = SPSA(model_wrap, sess=sess)
        attack_params = {'epsilon': attack_params_dict['eps'], 
                         'num_steps': attack_params_dict['n_steps']}
        batch_size = 1
    elif attack_params_dict['attack'] == 'deepfool':
        attack = DeepFool(model_wrap, sess=sess)
        attack_params = {'clip_min': 0., 'clip_max': 1.}
    elif attack_params_dict['attack'] == 'pgd':
        attack = ProjectedGradientDescent(model_wrap, sess=sess)
        attack_params = {'eps': attack_params_dict['eps'], 
                         'eps_iter': attack_params_dict['eps_iter'],
                         'nb_iter': attack_params_dict['n_steps'],
                         'clip_min': 0., 'clip_max': 1.}
    elif attack_params_dict['attack'] == 'carlini':
        attack = CarliniWagnerL2(model_wrap, sess=sess)
        attack_params = {'clip_min': 0., 'clip_max': 1.}
    else:
        raise NotImplementedError()

    return attack, attack_params, batch_size

    
def eval_cleverhans():

    # Set test phase
    learning_phase = K.learning_phase()
    K.set_learning_phase(0)

    # Pre-process images
    images_tf = images.astype(K.floatx())
    images_tf /= 255.

    # Wrapper for the Keras model
    model_wrap = KerasModelWrapper(model)

    # Initialize attack
    if attack_params_dict['attack'] == 'fgsm':
        attack = FastGradientMethod(model_wrap, sess=K.get_session())
        attack_params = {'eps': attack_params_dict['eps'], 'clip_min': 0., 
                         'clip_max': 1.}
    elif attack_params_dict['attack'] == 'deepfool':
        attack = DeepFool(model_wrap, sess=K.get_session())
        attack_params = {'clip_min': 0., 'clip_max': 1.}
    elif attack_params_dict['attack'] == 'madry':
        attack = ProjectedGradientDescent(model_wrap, sess=K.get_session())
        attack_params = {'clip_min': 0., 'clip_max': 1.}
    elif attack_params_dict['attack'] == 'carlini':
        attack = CarliniWagnerL2(model_wrap, sess=K.get_session())
        attack_params = {'clip_min': 0., 'clip_max': 1.}
    else:
        raise NotImplementedError()

    # Define input TF placeholder
    x = tf.placeholder(K.floatx(), shape=(None,) + images.shape[1:])
    y = tf.placeholder(K.floatx(), shape=(None,) + (labels.shape[-1],))

    # Define adversarial predictions symbolically
    x_adv = attack.generate(x, **attack_params)
    x_adv = tf.stop_gradient(x_adv)
    predictions_adv = model(x_adv)

    # Evaluate the accuracy of the model on adversarial examples
    eval_par = {'batch_size': batch_size}
    # feed_dict = {K.learning_phase(): attack_params_dict['learning_phase']}
    # acc_adv = model_eval(K.get_session(), x, y, predictions_adv, images, 
    #                      labels, feed=feed_dict, args=eval_par)
    acc_adv = model_eval(K.get_session(), x, y, predictions_adv, images_tf, 
                         labels, args=eval_par)

    print('Aversarial accuracy against %s: %.4f\n' %
          (attack_params_dict['attack'], acc_adv))

    # Set original phase
    K.set_learning_phase(learning_phase)

    return acc_adv


def _print_flags():
    # Print flags
    print('')
    print('--------------------------------------------------')
    print('Running %s' % sys.argv[0])
    print('')
    print('FLAGS:')
    flags = vars(FLAGS)
    for k in flags.keys():
        print('%s: %s' % (k, flags[k]))
    print('')


def _write_results(results, output_file):
    with open(output_file, 'w') as f:
        f.write('ADVERSARIAL EXAMPLES ROBUSTNESS:\n')
        with open(FLAGS.attack_params_file, 'r') as yml_file:
            cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
            for param in cfg:
                f.write('\t%s: %s\n' % (param, cfg[param]))
        f.write('\tTest accuracy on adversarial examples: %.4f\n' % results)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_file',
        type=str,
        default='/home/alex/git/research/projects/emotion/images/hdf5/cifar10.hdf5',
        help='Path to the HDF5 file containing the data set.'
    )
    parser.add_argument(
        '--group',
        type=str,
        default='cifar10_test',
        help='Group name in the HDF5 file indicating the test data set.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Number of images to process in a batch.'
    )
    parser.add_argument(
        '--image_params_file',
        type=str,
        default='/mnt/data/alex/git/research/projects/daug/daug_schemes/'
                'nodaug.yml',
        help='Path to the configuration file with the image pre-processing'
             'parameters'
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
        '--attack_params_file',
        type=str,
        default="attacks/fgsm_eps03.yml",
        help='Path to the configuration file with the attack parameters'
    )
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
        '--do_write',
        action='store_true',
        dest='do_write',
        help='Whether to write the results to a file'
    )
    parser.add_argument(
        '--log_device_placement',
        action='store_true',
        dest='log_device_placement',
        help='Enable log device placement.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
