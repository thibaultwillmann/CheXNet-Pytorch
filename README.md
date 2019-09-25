# Radiologist-Level Disese Detection on Chest X-Rays with Deep Learning 

Ofir Tayeb & Thibault Willmann

Submitted as a final project report for Deep Learning IDC, 2019

## 1. Introduction

Pneumonia is a disease in which the air sacs in one or both lungs get infected and inflame. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. Diseases such as Pneumonia are responsible for over 1 million hospitalizations and 50,000 deaths a year in the US alone. Currently radiologists use Chest X-Rays to detect diseases such as Pneumonia. Other diseases detected in this manner include Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pleural Thickening, Cardiomegaly, Nodule, Hernia, Mass. Once detected, the patient can be treated. However if the disease is not detected at an early stage, the consequences can be severe. 

![Image of Chest X-ray](https://github.com/thibaultwillmann/CheXNet-Pytorch/blob/master/data/images/00000013_005.png)
Example of a chest X-ray

Luckily Deep Learning networks can be trained to detect diseases and assist medical personel. In fact Deep Learning networks can be trained to detect diseases such as Pneumonia with greater accuracy than any human radiologist from chest X-Rays. Therfore, through decreasing human error in detection, countless lives can be saved!

Further an estimated two thirds of the global population lacks access to radiology diagnostics. These diagnostics include as mentioned above detection of diseases. With the automation of radiology experts, healthcare delivery can be improved and access to medical imaging expertise can be increased in many parts of the world. Therefore, through automating radiology experts, many parts of the world will gain radiology diagnostics and countless lives can be saved!

We set out to build an algorithm that could take as input a chest X-ray image and return probabilities for a collection of diseases detectable through chest X-rays (Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pleural Thickening, Cardiomegaly, Nodule, Hernia, Mass) and the probability of no disease being present. In Addition we envisioned the network returning a heat map of the original chest X-ray highlighting areas with high probalility of a disease being present.

![Image of Chest X-ray heatmap](https://github.com/thibaultwillmann/CheXNet-Pytorch/blob/master/heat_map.png)
Example of a chest X-ray heatmap

### 1.1. Related Works

The Stanford ML Group researched and came up with a Deep Learning Network to detect Pneumonia from chest X-Rays images.
The group detailed their findings in the paper [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays](https://arxiv.org/pdf/1711.05225.pdf) and features an [offical website](https://stanfordmlgroup.github.io/projects/chexnet/). 

The group decided to use the 121 layer *DenseNet* convolutional neural network, taking advantage of each layer obtaining additional inputs from all preceding layers and passing on its own feature-maps to all subsequent layers. Basically each layer is receiving a “collective knowledge” from all preceding layers.

![Image of one DenseNet Block](https://github.com/thibaultwillmann/CheXNet-Pytorch/blob/master/denseNet_block.png)

The dataset the group used to train their network, was released by the US National Institute of Health and contains 112,120 frontal-view X-ray images of 30,805 unique patients, annotated with up to 14 different thoracic pathology labels using NLP methods on radiology reports. They labeled images that have pneumonia as one of the annotated pathologies as positive examples and label all other images as negative examples for the pneumonia detection task.

The network receives a chest X-Rays image and output the probability of Pneumonia being present together with a chest X-Ray heatmap highlighting areas with high probalility of Pneumonia being present.

# 2. Solution

## 2.1. General Approach

how you want to solve it, what alternative you plan to try and why

## 2.2. Design

We are using Pytorch, which is an open source machine learning library used mainly for Deep Learning tasks such as Computer Vision and Natural Language Processing. Pytorch was developed by Facebooks Artificial Intelligence Research Group under Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan. 

Our model consists of the pytorch implementation of the DenseNet convolutional neural network with 121 layers available under the torchvision library and an additional fully connected linear layer. 

The network expects an image of dimension [channel, height, width], we are using [3,244,244].

The model has an output layer of 15 neurons, each indicating the probability of Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Hernia, Mass, No Finding (no disease present) accordingly. The output is passed as a FloatTensor.

information about the platform, code, how long to train, technical challenges

# References
DenseNet [https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

Stanford ML Group (CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays) [https://stanfordmlgroup.github.io/projects/chexnet/](https://stanfordmlgroup.github.io/projects/chexnet/)
