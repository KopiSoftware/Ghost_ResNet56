# Ghost_ResNet56
This is an experimental Pytorch implementation of Ghost-Resnet56. I finished the adaption follows [iamhankai](https://github.com/iamhankai/ghostnet.pytorch)’s and [akamaster](https://github.com/akamaster/pytorch_resnet_cifar10)’s work. 


- **Weights** Just as the paper [arxiv](https://arxiv.org/abs/1911.11907) shows, the number of the parameters of the adapted resnet56 decreased from 0.85m to 0.44m.
- **Training** I still working on it. Now we can train the Ghostnet and the Ghost Resnet56 on the Cifar-10 dataset, but I cannot obtain the. same performance on both models. I have to follow this paper (**[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.**) to achieve the same results. Besides, I have some ideas about it.
- **Somethng interesting** This implements including Ghostnet can be trained on the ipad. You probably have to download an app called Python AI, which pre-installed the pytorch arm version in it and makes compilation possible.  The trained weights take about 20mb.
