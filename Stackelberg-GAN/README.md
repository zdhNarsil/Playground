# Stackelberg-GAN
This is the code for the paper "Stackelberg GAN: Towards Provable Minimax Equilibrium via Multi-Generator Architectures".

The code is written in python and requires numpy, matplotlib, torch, torchvision and the tqdm library.

## Install
This code depends on python 3.6, pytorch 0.4.1. We suggest to install the dependencies using Anaconda or Miniconda. Here is an exemplary command:
```
$ wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
$ bash Anaconda3-5.1.0-Linux-x86_64.sh
$ source ~/.bashrc
$ conda install pytorch=0.4.1
```

## Get started
To get started, cd into the directory. Then runs the scripts: 
* gan_stackelberg_mG.py is a demo on the performance of Stackelberg GAN on Gaussian mixture dataset,
* gan_branch_mG.py is a demo on the performance of multi-branch GAN (a baseline method) on Gaussian mixture dataset,
* gan_mnist_classifier.py is a demo on the performance of Stackelberg GAN on MNIST dataset,
* gan_mnist_fashion_classifier.py is a demo on the performance of Stackelberg GAN on fashion-MNIST dataset.

## Using the code
The command `python xxx.py --help` gives the help information about how to run the code.

## Architecture of Stackelberg GAN

Stackelberg GAN is a general framework which can be built on top of all variants of standard GANs. The key idea is to apply multiple generators which team up to play against the discriminator.

<p align="center">
    <img src="images/architecture.png" width="500"\>
</p>



## Experimental Results

### Mixture of Gaussians

<p align="center">
    <img src="images/converge.gif" width="400"\>
</p>

We test the performance of varying architectures of GANs on a synthetic mixture of Gaussians dataset with 8 modes and 0.01 standard deviation. We observe the following phenomena:

*Naïvely increasing capacity of one-generator architecture does not alleviate mode collapse*. It shows
that the multi-generator architecture in the Stackelberg GAN effectively alleviates the mode collapse issue.
Though naïvely increasing capacity of one-generator architecture alleviates mode dropping issue, for more
challenging mode collapse issue, the effect is not obvious.

#### Running Example
<p align="center">
    <img src="images/exp1.png" width="600"\>
</p>

*Stackelberg GAN outperforms multi-branch models.* We compare performance of multi-branch GAN (i.e., classic GAN with multi-branch architecture for its generator) and Stackelberg GAN. The performance of Stackelberg GAN is also better than multi-branch GAN of much larger capacity.

#### Running Example
<p align="center">
    <img src="images/exp2.png" width="600"\>
</p>

*Generators tend to learn balanced number of modes when they have same capacity*. We observe that
for varying number of generators, each generator in the Stackelberg GAN tends to learn equal number of
modes when the modes are symmetric and every generator has same capacity.

#### Running Example
<p align="center">
    <img src="images/exp3.png" width="600"\>
</p>

### MNIST Dataset
The following figure shows the diversity of generated digits by Stackelberg GAN with varying number of generators. *Left Figure:*
Digits generated by the standard GAN. It shows that the standard GAN generates many "1"’s which are not very diverse. *Middle Figure:* Digits generated by the Stackelberg GAN with 5 generators, where every two rows correspond to one generator. *Right Figure:* Digits generated by the Stackelberg GAN with 10 generators, where each row corresponds to one generator. As the number of generators increases, the images tend to be more diverse.

#### Running Example
<p align="center">
    <img src="images/mnist.png" width="600"\>
</p>

### Fashion-MNIST Dataset
The following figure shows the diversity of generated fashions by Stackelberg GAN with varying number of generators. *Left Figure:*
Examples generated by the standard GAN. It shows that the standard GAN fails to generate bags. *Middle Figure:* Examples generated by the Stackelberg GAN with 5 generators, where every two rows correspond to one generator. *Right Figure:* Examples generated by the Stackelberg GAN with 10 generators, where each row corresponds to one generator.

#### Running Example
<p align="center">
    <img src="images/fashion_mnist.png" width="600"\>
</p>

## Reference
For technical details and full experimental results, see [the paper](https://arxiv.org/abs/1811.08010).
```
@article{Zhang2018stackelberg, 
	author = {Hongyang Zhang and Susu Xu and Jiantao Jiao and Pengtao Xie and Ruslan Salakhutdinov and Eric P. Xing}, 
	title = {Stackelberg GAN: Towards Provable Minimax Equilibrium via Multi-Generator Architectures}, 
	journal={arXiv preprint arXiv:1811.08010},
	year = {2018}
}
```

## Contact
Please contact hongyanz@cs.cmu.edu if you have any question on the codes.