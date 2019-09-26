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

Our task is to build an algorithm that for a given chest X-Ray image returns probabilities for different diseases (Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pleural Thickening, Cardiomegaly, Nodule, Hernia, Mass) being present and the probability of no disease being present. We use a convolutional neural network to solve this task. CNNs are Deep Learning algorithms which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.
Therefore idealy CNNs will recognize small features in the first layers and larger features in the later layers. Using the learned features the CNN will be able to distinguish between the different diseases.

## 2.2. Design

show code examples!!!!!!

### 2.2.1. Model

As a model we use a 121 layer *DenseNet* convoluted neural network. We use a DenseNet, because they improve flow of information and gradients through the network. Thus they make the optimization of very deep networks easy to control. The weights of the network are initialized with weights from a model pretrained on [ImageNet](http://image-net.org). We add a final fully connected layer with 15 neuron outputs. Finally we apply a sigmoid nonlinearity function on each neuron. Each output will indicate the probability of a certain disease (Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pleural Thickening, Cardiomegaly, Nodule, Hernia, Mass) being present with the last output returning the probability of no disease being present.

We are using Pytorch, which is an open source machine learning library used mainly for Deep Learning tasks such as Computer Vision and Natural Language Processing. Pytorch was developed by Facebooks Artificial Intelligence Research Group under Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan. Our model consists of the pytorch implementation of the DenseNet convolutional neural network with 121 layers available under the torchvision library and an additional fully connected linear layer. 

The network expects an image of dimension [channel, height, width], we are using [3,244,244]. The output is passed as a FloatTensor with 15 entries.

### 2.2.2. Data

As the data to train and test the network we use the public and free data set *NIH Chest X-ray Dataset* on [Kaggle](https://www.kaggle.com/nih-chest-xrays/data). The National Insitute of Health (NIH) Chest X-ray Dataset is comprised of 112,120 X-Ray images with disease labels from 30,805 unique patients. To create these labels, Natural Language Processing to text-mine disease classifications from the associated radiological reports was used. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. 

There are 15 classes (14 diseases, and one for "No findings"). Images can be classified as "No findings" or one or more disease classes:
- Atelectasis
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural_thickening
- Cardiomegaly
- Nodule Mass
- Hernia

The images are of size 1024 x 1024.

Problems to note about the data: 
- The image labels are NLP extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%. 
- Chest X-Ray radiology reports are not anticipated to be publicly shared. Parties who use this public dataset are encouraged to share their “updated” image labels through manual annotation

We randomly sampled 5% of these images and created a smaller dataset. The random sample contains 5606 X-Ray images and class labels. The X-Ray images are stored in *data/images/* and the class labels in *data/sample_labels.csv*. Each row in *data/sample_labels.csv* has the format 
```
00000013_005.png,Emphysema|Infiltration|Pleural_Thickening|Pneumothorax, ...
```

### 2.2.3. Training

optimizer
loss function
how long

# References
DenseNet [https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

Stanford ML Group (CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays) [https://stanfordmlgroup.github.io/projects/chexnet/](https://stanfordmlgroup.github.io/projects/chexnet/)
