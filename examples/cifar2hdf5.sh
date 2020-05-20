# This script downloads CIFAR-10 data set and stores it as a HDF5 file
#
# This script is meant to to be run from the root of the repository
# Consider changing the paths to files and directories
#
python datasets/cifar2hdf5.py \
--output_file ~/datasets/hdf5/cifar10.hdf5 \
--download_dir ~/datasets/
