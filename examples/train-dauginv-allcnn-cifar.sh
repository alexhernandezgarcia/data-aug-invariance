# This script trains the All-CNN architecture on CIFAR-10 with data augmentation invariance
#
# This script is meant to to be run from the root of the repository
# Consider changing the paths to files and directories
#
# CIFAR-10
#
# All-CNN
#
# no-reg bn heavier inv
python train.py \
--data_file ~/datasets/hdf5/cifar10.hdf5 \
--group_tr train \
--group_val test \
--train_dir ./log/cifar10/allcnn/noreg/bn/inv/r0/ \
--daug_params heavier.yml \
--train_config_file config_train/cifar/sgd/allcnn/noreg_bn.yml \
--test_config config_test/daug_quick.yml \
--aug_per_img_tr 8 \
--daug_invariance_params loss01_exp.yml \
--class_invariance_params noinv.yml \
--save_model_every 0
