# Ghost_ResNet56
This is an experimental Pytorch implementation of the Ghost-Resnet56. I finish the adaption follow [iamhankai](https://github.com/iamhankai/ghostnet.pytorch)’s and [akamaster](https://github.com/akamaster/pytorch_resnet_cifar10)’s work. 


- **Weights** Just as the paper [arxiv](https://arxiv.org/abs/1911.11907) describes, the number of the parameters of the adapted resnet56 decreased from 0.85m to 0.44m.
- **Training** Now we can train the Ghostnet and the Ghost Resnet56 on the Cifar-10 dataset, but I cannot obtain the same performance on both models. I have to follow this paper (**[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.**) to achieve the same results.
- **Somethng interesting** This implements including Ghostnet can be trained on the ipad. You probably have to download an app called "Python AI", which pre-installed the pytorch arm version in it and makes compilation possible.  The trained weights of ghostnet take about 20mb.


### How to train models
Simply run the "train_Ghost_ResNet56.py". 
  - If you want to try different optimization strategy, modify the milestone in Ln88 or uncommen the Ln89 to switch to the adam optimizer.
  - Uncommen Ln142, it will beep during each epochs.
