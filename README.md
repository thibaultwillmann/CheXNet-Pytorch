# Radiologist-Level Disease Detection on Chest X-Rays with Deep Learning 

Ofir Tayeb & Thibault Willmann

Submitted as a final project report for Deep Learning IDC, 2019

## 1. Introduction

Pneumonia is a disease in which the air sacs in one or both lungs get infected and inflame. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. Diseases such as Pneumonia are responsible for over 1 million hospitalizations and 50,000 deaths a year in the US alone. Currently radiologists use Chest X-Rays to detect diseases such as Pneumonia. Other diseases detected in this manner include Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pleural Thickening, Cardiomegaly, Nodule, Hernia and Mass. Once detected, the patient can be treated. However if the disease is not detected at an early stage, the consequences can be severe. 

Luckily algorithms can be trained to detect diseases and assist medical personel. In fact algorithms can be trained to detect diseases such as Pneumonia with greater accuracy than any human radiologist from chest X-Rays. Therfore, through decreasing human error in detection, countless lives can be saved!

Further an estimated two thirds of the global population lacks access to radiology diagnostics. These diagnostics include as mentioned above detection of diseases. With the automation of radiology experts, healthcare delivery can be improved and access to medical imaging expertise can be increased in many parts of the world. Therefore, through automating radiology experts, many parts of the world will gain radiology diagnostics and countless lives can be saved!

We set out to build an algorithm that could take as input a chest X-ray image and return probabilities for a collection of diseases detectable through chest X-rays (Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pleural Thickening, Cardiomegaly, Nodule, Hernia, Mass) and the probability of no disease being present. 

![Image of chest X-Ray and heatmap](https://github.com/thibaultwillmann/CheXNet-Pytorch/blob/master/chest_x_ray_example.png)

Image of a chest X-Ray left and heatmap highlighting areas with high probalility of a disease being present right

### 1.1. Related Works

The Stanford ML Group researched and came up with a Deep Learning Network to detect Pneumonia from chest X-Rays images.
The group detailed their findings in the paper [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays](https://arxiv.org/pdf/1711.05225.pdf) and features an [offical website](https://stanfordmlgroup.github.io/projects/chexnet/). 

The network receives a chest X-Rays image and output the probability of Pneumonia being present together with a chest X-Ray heatmap highlighting areas with high probalility of Pneumonia being present.

# 2. Solution

## 2.1. General Approach

Our task is to build an algorithm that for a given chest X-Ray image returns probabilities for different diseases (Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pleural Thickening, Cardiomegaly, Nodule, Hernia, Mass) being present and the probability of no disease being present. We use a convolutional neural network to solve this task. CNNs are Deep Learning algorithms which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.
Therefore CNNs will idealy recognize small features in the first layers and larger features in the later layers. Using the learned features, the CNN will be able to distinguish between the different diseases.

## 2.2. Design

### 2.2.1. Model

We are using Pytorch, which is an open source machine learning library used mainly for Deep Learning tasks such as Computer Vision and Natural Language Processing. Pytorch was developed by Facebooks Artificial Intelligence Research Group under Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan.

As a model we use a 121 layer *DenseNet* convoluted neural network. 

![DenseNet 121](https://github.com/thibaultwillmann/CheXNet-Pytorch/blob/master/densenet.png)

DenseNet 121

![DenseNet Block](https://github.com/thibaultwillmann/CheXNet-Pytorch/blob/master/densenet_block.png)

DenseNet Block

We use a DenseNet, because they improve flow of information and gradients through the network. Thus they make the optimization of very deep networks easy to control. For intuition sake, we take advantage of each layer obtaining additional inputs from all preceding layers and passing on its own feature-maps to all subsequent layers. Therefore each layer is receiving a “collective knowledge” from all preceding layers. The weights of the network are initialized with weights from a model pretrained on [ImageNet](http://image-net.org). We use the pytorch implementation of the *DenseNet* CNN available under the torchvision library. We add a final fully connected layer with 15 neuron outputs. Finally we apply a sigmoid nonlinearity function on each neuron. 
```
class DenseNet121(nn.Module):
    
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
```

Each output will indicate the probability of a certain disease (Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pleural Thickening, Cardiomegaly, Nodule, Hernia, Mass) or probability of no disease being present in the input image.

The network expects an image of dimension channel x height x width, we are using 3 x 244 x 244. The output is passed as a FloatTensor with 15 entries.

### 2.2.2. Data

As the data to train and test the network we use the public and free data set *NIH Chest X-ray Dataset* on [Kaggle](https://www.kaggle.com/nih-chest-xrays/data). The National Insitute of Health (NIH) chest X-Ray dataset is comprised of 112,120 X-Ray images with disease labels from 30,805 unique patients. To create these labels, Natural Language Processing to text-mine disease classifications from the associated radiological reports was used. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. 

There are 15 classes (14 diseases, and one for "No findings"). Images can be classified as *No findings* or one or more disease classes:
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

We encode each class label as a FloatTensor of length 15 for the model. Each disease in the disease_list of a single instance is weighted with 1:
```
classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
}

labelTensor = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
for disease in diseases_list:
        labelTensor = labelTensor.add(classEncoding[disease])
```

We apply some preprocessing on each greyscale image. The image is converted to RGB and resized to 256x256. Then ten crops of size 224 x 224 are generated consisting of the four corners and the center plus the horizontal flipped version of these. These are transformed to a tensor and normalized. Finally the image has a dimension of 10 x 3 x 224 x 224 containing obviously ten crops.
```
image = Image.open(image_path).convert('RGB')
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.TenCrop(224),
  transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
  transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
])
image = preprocess(image)
```

### 2.2.3. Training

The weights of the network are initialized with weights from a model pretrained on ImageNet (Deng et al., 2009). The network is trained end-to-end using Adam.
We train the model using mini- batches of size 5, use an initial learning rate of 0.001, binary cross entropy loss function and stochastic gradient descent optimizer.

```
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

We randomly split the dataset into training set of 4485 images (80%), and test set of 1121 images. There is no patient overlap between the sets.

```
trainloader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=True, num_workers=5)
```

In total the training process took 4 hrs and 20 min with 69% accuracy on the training set.

```
Epoch: 15, loss: 131.393, Accuracy: 69.030

CPU times: user 2h 21min 44s, sys: 1h 13min 31s, total: 3h 35min 16s
Wall time: 4h 18min 28s
```

# 3. Experimental Results

ChexNet outputs a vector t of binary labels indicating the absence or presence of each of the following 14 pathology classes: Atelec- tasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nod-ule, Pleural Thickening, Pneumonia, and Pneumotho- rax. We replace the final fully connected layer in CheXNet with a fully connected layer producing a 15-dimensional output, after which we apply an elementwise sigmoid nonlinearity. The final output is the predicted probability of the presence of each pathology class. 

We find that CheXNet achieves results of 53.8% accuracy on the test set. 

```
correct = 0
total = 0
with torch.no_grad():
  for i, (images, labels) in enumerate(testloader, 0):
    images = images.cuda()
    n_batches, n_crops, channels, height, width = images.size()
    image_batch = torch.autograd.Variable(images.view(-1, channels, height, width))
    labels = tile(labels, 0, 10).cuda()
    outputs = model(image_batch)
    correct += compute_score_with_logits(outputs, labels).sum()
    total += labels.size(0)
    

print('Accuracy on test set: %.3f' % (100 * correct / total))
```
```
Accuracy on test set: 53.872
```

# 4. Discussion

Pneumonia is a major cause of patients dying in hospitals today. To prevent death, a early detection and treatment of pneumonia is critical. Chest X-rays are the most common examination tool used in practice with 2 billion made a year.
However, two thirds of the global population lacks access to radiology diagnostics. In Addition, even when the equipment is available, experts who can interpret X-rays are often missing.

Therefore we developed an algorithm which detects diseases such as pneumonia from front chest X-ray images. Since we only had access to the very limited computing power of Colab, we were limited to 5% of the chest X-ray images and only few epochs. However if more computing power was at our disposal we could achieve a level of accuracy exceeding practicing radiologists. In conclusion this algorithm can and should save lives in many parts of the world by assisting medical staff which lacks skilled radiologists or assist radiologists directly.

# 5. Code

Find the entire code [here](https://github.com/thibaultwillmann/CheXNet-Pytorch/blob/master/model.py).
The colab notebook used to train the network can be found [here](https://github.com/thibaultwillmann/CheXNet-Pytorch/blob/master/CheXnet.ipynb).

# References

- DenseNet [https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

- Stanford ML Group (CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays) [https://stanfordmlgroup.github.io/projects/chexnet/](https://stanfordmlgroup.github.io/projects/chexnet/)

- CheXNet Paper [https://arxiv.org/pdf/1711.05225.pdf](https://arxiv.org/pdf/1711.05225.pdf)

- NIH Chest X-ray Dataset [https://www.kaggle.com/nih-chest-xrays/data](https://www.kaggle.com/nih-chest-xrays/data)

- ImageNet Dataset [http://image-net.org](http://image-net.org)
