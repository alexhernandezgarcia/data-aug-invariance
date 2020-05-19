from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tarfile
from six.moves import urllib

import h5py

import tensorflow as tf
import numpy as np
import pickle

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck']

FLAGS = None
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_DEPTH = 3
N_CLASSES = 10
URL_DATA = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def main(argv=None):

    # Download CIFAR-10 data set
    files = _download_cifar(FLAGS.download_dir)

    # Get train, validation and test sets
    images_train, labels_train, filenames_train = _read_data(files, 'data')
    images_test, labels_test, filenames_test = _read_data(files, 'test')
    labels_train = np.asarray(labels_train)
    labels_test = np.asarray(labels_test)

    images_train, labels_train, filenames_train, \
    images_val, labels_val, filenames_val = _split_train_val(
            images_train, labels_train, filenames_train, FLAGS.pct_val) 

    # Get number of examples in train and test sets
    num_train = images_train.shape[0]
    num_val = images_val.shape[0]
    num_test = images_test.shape[0]

    # Convert labels to one-hot encoding
    labels_one_hot_tr = np.zeros([num_train, N_CLASSES], dtype=int)
    labels_one_hot_tr[np.arange(num_train), labels_train] = 1
    labels_one_hot_val = np.zeros([num_val, N_CLASSES], dtype=int)
    labels_one_hot_val[np.arange(num_val), labels_val] = 1
    labels_one_hot_tt = np.zeros([num_test, N_CLASSES], dtype=int)
    labels_one_hot_tt[np.arange(num_test), labels_test] = 1

    # Set up the parameters for class imbalance
    if FLAGS.class_imbalance == 'step':
        n_cl_minority = int(FLAGS.imbalance_mu * N_CLASSES)
        max_n = int(num_train / N_CLASSES)
        n_minority = np.repeat(int(max_n / FLAGS.imbalance_rho), n_cl_minority)
        minority_classes = np.random.permutation(N_CLASSES)[:n_cl_minority]
    elif FLAGS.class_imbalance == 'linear':
        minority_classes = np.random.permutation(N_CLASSES - 1)
        max_n = int(num_train / N_CLASSES)
        min_n = int(max_n / FLAGS.imbalance_rho)
        n_minority = np.asarray(np.interp(x=np.arange(2, N_CLASSES),
                                          xp=[1, N_CLASSES], 
                                          fp=[min_n, max_n]), 
                                dtype=int)
        n_minority = np.r_[min_n, n_minority]
    elif FLAGS.class_imbalance == 'balanced':
        pass
    else:
        raise ValueError('Type of imbalance must be step, linear or balanced')

    # Apply class imbalance by removing examples from the arrays
    if FLAGS.class_imbalance != 'balanced':
        for c, n in zip(minority_classes, n_minority):
            print(c, n)
            idx = np.where(labels_one_hot_tr[:, c] == 1)[0]
            idx_del = np.random.permutation(idx)[n:]
            images_train = np.delete(images_train, idx_del, axis=0)
            labels_one_hot_tr = np.delete(labels_one_hot_tr, idx_del, axis=0)
            labels_train = np.delete(labels_train, idx_del, axis=0)
        num_train = images_train.shape[0]
            
    # Open HDF5 file
    with h5py.File(FLAGS.output_file, 'w') as hdf5_file:

        # Create the Groups to store the Datasets
        grp_tr = hdf5_file.create_group('train')
        grp_tt = hdf5_file.create_group('test')

        if num_val > 0:
            grp_val = hdf5_file.create_group('val')

        # Create the datasets that will contain the image data, labels and ids
        data_tr = grp_tr.create_dataset('data', shape=(
            num_train, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), dtype=np.uint8)
        labels_tr_h5 = grp_tr.create_dataset('labels', shape=(
            num_train, N_CLASSES), dtype=np.uint8)
        ids_tr = grp_tr.create_dataset('ids', shape=(num_train, 2), 
                                       dtype=h5py.special_dtype(vlen=str))

        data_tt = grp_tt.create_dataset('data', shape=(
            num_test, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), dtype=np.uint8)
        labels_tt_h5 = grp_tt.create_dataset('labels', shape=(
            num_test, N_CLASSES), dtype=np.uint8)
        ids_tt = grp_tt.create_dataset('ids', shape=(num_test, 2), 
                                       dtype=h5py.special_dtype(vlen=str))

        if num_val > 0:
            data_val = grp_val.create_dataset('data', shape=(
                num_val, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), dtype=np.uint8)
            labels_val_h5 = grp_val.create_dataset('labels', shape=(
                num_val, N_CLASSES), dtype=np.uint8)
            ids_val = grp_val.create_dataset('ids', 
                    shape=(num_val, 2), dtype=h5py.special_dtype(vlen=str))

        # Permute the indices in order to shuffle the images in the HDF5 file
        if FLAGS.shuffle:
            indices_tr = np.random.permutation(num_train)
            indices_val = np.random.permutation(num_val)
            indices_tt = np.random.permutation(num_test)
        else:
            indices_tr = range(num_train)
            indices_val = range(num_val)
            indices_tt = range(num_test)

        # Fill data
        data_tr[:, :, :, :] = images_train[indices_tr, :, :, :]
        if FLAGS.shuffle_train_labels:
            rand_indices_tr = np.random.permutation(indices_tr)      
            labels_tr_h5[:, :] = labels_one_hot_tr[rand_indices_tr, :]
            ids_tr[:, 0] = [str(filenames_train[i]) for i in rand_indices_tr]
            ids_tr[:, 1] = [classes[cl] for cl in labels_train]
        else:
            labels_tr_h5[:, :] = labels_one_hot_tr[indices_tr, :]
            ids_tr[:, 0] = [str(filenames_train[i]) for i in indices_tr]
            ids_tr[:, 1] = [classes[cl] for cl in labels_train]

        data_tt[:, :, :, :] = images_test[indices_tt, :, :, :]
        labels_tt_h5[:, :] = labels_one_hot_tt[indices_tt, :]
        ids_tt[:, 0] = [str(filenames_test[i]) for i in indices_tt]
        ids_tt[:, 1] = [classes[cl] for cl in labels_test]

        if num_val > 0:
            data_val[:, :, :, :] = images_val[indices_val, :, :, :]
            labels_val_h5[:, :] = labels_one_hot_val[indices_val, :]
            ids_val[:, 0] = [str(filenames_val[i]) for i in indices_val]
            ids_val[:, 1] = [classes[cl] for cl in labels_val]


def _download_cifar(download_dir):
    """
    Downloads and extracts the CIFAR-10 data set from the URL.

    Parameters
    ----------
    download_dir : str
        The path were the tar.gz file is stored and the files extracted

    Returns
    -------
    list str
        A list containing the file structure of the extracted elements.
    """

    filename = URL_DATA.split('/')[-1]
    filepath = os.path.join(download_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading {} {:.f} %'.format(
                filename, 
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(URL_DATA, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    print('Extracting %s...' % filename)
    tarfile.open(filepath, 'r:gz').extractall(download_dir)

    return tarfile.open(filepath, 'r:gz').getnames()


def _read_data(files, keyword):
    """
    Reads the binary extracted files and return single matrices for the
    specified set.

    Parameters
    ----------
    files : list str
        A list containing the file structure of extracted elements.

    keyword : str
        Keyword to specify the set (train, test --> 'data', 'test')

    Returns
    -------
    images :  ndarray
        An array containing all the available training images from CIFAR-10,
        with shape [N, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH]

    labels : list int
        List of labels

    filenames : lsit str
        List of filenames

    """

    # Get paths of train batches
    files = [s for s in files if s.find(keyword) != -1]

    if not files:
        raise ValueError('There are no extracted files that match the '
                         'specified keyword ({})'.format(keyword))

    # Read files
    images = []
    labels = []
    filenames = []
    for path in files:
        with open(os.path.join(FLAGS.download_dir, path), 'rb') as f:
            train_dict = pickle.load(f, encoding='bytes')

            images.append(train_dict[b'data'])
            labels.append(np.array(train_dict[b'labels']))
            filenames.append(np.array(train_dict[b'filenames']))

    # Concatenate lists into ndarrays
    images = np.concatenate(images, axis=0)
    labels = [l for labels_batch in labels for l in labels_batch]
    filenames = [f for filenames_batch in filenames for f in filenames_batch]

    # Reshape images
    images = np.reshape(images, 
                        [images.shape[0], IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH])
    images = images.transpose(0, 2, 3, 1)

    return images, labels, filenames


def _split_train_val(images, labels, filenames, pct_val, shuffle=True):

    n_total = images.shape[0]
    n_tr = n_total - int(pct_val * n_total)

    if shuffle:
        indices = np.random.permutation(n_total)
    else:
        indices = range(n_total)

    indices_tr = indices[:n_tr]
    indices_val = indices[n_tr:]

    images_tr = images[indices_tr, :, :, :]
    labels_tr = labels[indices_tr]
    filenames_tr = [filenames[idx] for idx in indices_tr]

    images_val = images[indices_val, :, :, :]
    labels_val = labels[indices_val]
    filenames_val = [filenames[idx] for idx in indices_val]

    return images_tr, labels_tr, filenames_tr, \
           images_val, labels_val, filenames_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_file',
        type=str,
        default='/home/alex/tmp/cifar10.hdf5',
        help='Output HDF5 (.hdf5) file'
    )
    parser.add_argument(
        '--download_dir',
        type=str,
        default='/tmp/cifar10/',
        help='Path where the temporary CIFAR 10 dataset is downloaded'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        dest='shuffle',
        help='True to re-shuffle the train, test and validation partitions'
    )
    parser.add_argument(
        '--shuffle_train_labels',
        action='store_true',
        dest='shuffle_train_labels',
        help='True to shuffle the labels'
    )
    parser.add_argument(
        '--pct_val',
        type=float,
        default=0.,
        help='The percentage of training images that should be used to create '
             'a specific validation set (group)'
    )
    parser.add_argument(
        '--class_imbalance',
        type=str,
        default='balanced',
        help='Type of class imbalance. Must be one of: balanced, step, linear'
    )
    parser.add_argument(
        '--imbalance_mu',
        type=float,
        default=0.5,
        help='The percentage of minority classes in step class imbalance, as '
             'defined by Buda et al. (2018)'
    )
    parser.add_argument(
        '--imbalance_rho',
        type=int,
        default=2,
        help='The ratio between the number of examples in the majority classes'
             ' and the minority classes, as defined by Buda et al. (2018)'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
