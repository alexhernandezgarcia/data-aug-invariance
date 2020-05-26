# Data augmentation and invariance training

This repository contains part of the code of research projects developed by [Alex Hernandez-Garcia](https://alexhernandezgarcia.github.io/), focused on comparing the impact of [data augmentation and explicit regularisation](https://arxiv.org/abs/1806.03852) in deep neural networks trained for image object categorisation, as well as the implementation of perceptually and biologically inspired methods, such as [data augmentation invariance](https://arxiv.org/abs/1906.04547).

## Disclaimer about versions
Please note that the current version has not been tested with the most up to date version of the libraries. In particular, the functionality has been tested with Keras 2.1.5 and TensorFlow 1.4.0. I am currently working on upgrading the framework to Keras 2.3 and TensorFlow 2. 

## Usage
Have a look at the shell scripts in [examples](./examples) to find out some of the options that this project offers. Note that one of the main features is the flexibility to easily change aspects of the training process (regularisation, data augmentation, invariance training, hyperparameters, evaluation) through arguments and configuration files.

## Citation

If you use this code for scientific purposes, please consider citing:

*Data augmentation instead of explicit regularization. Alex Hernandez-Garcia, Peter König, 2018. arXiv:1806.03852*

	@article{hergar2018daugreg,
		author = {Hernandez-Garcia, Alex and K{\"o}nig, Peter},
		title = {Data augmentation instead of explicit regularization},
        journal = {arXiv preprint arXiv:1806.03852},
		year = {2018}
	}

*Learning robust visual representations using data augmentation invariance. Alex Hernandez-Garcia, Peter König, Tim C. Kietzmann, 2019. arXiv:1906.04547*

	@article{hergar2019dauginv,
		author = {Hernandez-Garcia, Alex and K{\"o}nig, Peter and Kietzmann, Tim},
		title = {Data augmentation instead of explicit regularization},
        journal = {arXiv preprint arXiv:1906.04547},
		year = {2019}
	}

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
