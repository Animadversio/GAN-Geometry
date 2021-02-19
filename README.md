# "A Geometric Analysis of Deep Generative Image Models and Its Applications" Official Code 
 
This repo curate generic tools for **analyzing the latent geometry of generative models**, and using the goemetric information to improve on various applications like GAN interpretability, inversion, optimization in latent space. Specifically we can compute the Riemannian metric tensor of the latent space, pulling back certain image distance function. 

A work published in ICLR 2021. 

* [Open Review](https://openreview.net/forum?id=GH7QRzUDdXG)
* [Arxiv](https://arxiv.org/abs/2101.06006)

![](img\title_img.png)

## How it works?


## Structure of Repo
`analysis` contains code for analyzing computed Hessian information and generate figure and statistics from it. 

## Key Dependency

pytorch (1.5.0, py3.7_cuda101_cudnn7_0)
scipy
CUDA (10.1)

For [`hessian-eigenthings`](https://github.com/noahgolmant/pytorch-hessian-eigenthings) installation, use this 
`pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings`

For [LPIPS](https://github.com/richzhang/PerceptualSimilarity)

Code has been tested on GTX 1060 GPU (6GB). 


### Obtain pre-trained GANs
Our algorithm is a generic analysis that could be applied to generative models. To repreoduce the results in the paper, you need to obtain some pre-trained GANs. 

* [DCGAN](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_dcgan/). Trained on 64 by 64 pixel fashion dataset. It has a 120d latent space, using Gaussian as latent space distribution. 
* [Progressive Growing GAN (PGGAN)](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/). 
* [DeePSim/FC6GAN](https://lmb.informatik.uni-freiburg.de/people/dosovits/code.html). This model is based on DCGAN architechture. We translated it into pytorch, included the model definition in the script and hosted the weights. 
* [BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN). From HuggingFace. We used 256 pix version in the paper, with 128d noise vector input and 128d class embedding input. 
* [BigBiGAN](https://tfhub.dev/deepmind/bigbigan-resnet50/1). Weights obtained from Deepmind official tf version. The generator could be translated into pytorch. We used bigbigan-resnet50 version, with 120d latent space and 128 pix output.  
* [StyleGAN](https://github.com/rosinality/style-based-gan-pytorch). We used 256 pix version, with 512d Z and W latent space. 
* [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch). We translated the weights of pretrained models from [this list](https://pythonawesome.com/a-collection-of-pre-trained-stylegan-2-models-to-download), to pytorch. All with 512d Z and W latent space, with various spatial resolution. 
* [WaveGAN](https://github.com/mostafaelaraby/wavegan-pytorch/). An audio generating GAN. We trained it ourselves using piano dataset. 

To analyze your own GAN, follow this tutorial to come....

## Efficiency of Analysis