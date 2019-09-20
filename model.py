#data from https://www.kaggle.com/nih-chest-xrays/data

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image

class DenseNet121(nn.Module):
    
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1000, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
    def save_param(self):
        torch.save(self.model.state_dict(), "tensor.pt")

    def load_param(self):
        self.model.load_state_dict(torch.load("tensor.pt"))

class DataPreprocessing(Dataset):

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

    def __init__(self,classEncoding=classEncoding):
        self.image_names = []
        self.labels = []
        with open("data/sample_labels.csv", "r") as f:
            title = True
            for line in f:
                if (title):
                    title = False
                    continue

                items = line.split(",")
                image_name = items[0]
                image_name = os.path.join("data/images/", image_name)
                self.image_names.append(image_name)

                label = items[1]  # list of diseases
                diseases_list = label.split("|")
                labelTensor = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                for disease in diseases_list:
                    labelTensor = labelTensor.add(classEncoding[disease])
                self.labels.append(labelTensor)

    def __getitem__(self, index):
        image_path = self.image_names[index]
        image = Image.open(image_path).convert('RGB')
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
        image = preprocess(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_names)


class Train():

    def __init__(self, trainset, model):

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(1):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (images, labels) in enumerate(trainloader, 0): # get the inputs; data is a list of [images, labels]

                #images.shape -> [64, 10, 3, 224, 224]
                #labels.shape -> [64, 15]

                # zero the parameter gradients
                optimizer.zero_grad()

                #format input
                n_batches, n_crops, channels, height, width = images.size()
                image_batch = torch.autograd.Variable(images.view(-1, channels, height, width)) #640 images: 64 batches contain 10 crops each decomposed into 640 images

                labels = self.tile(labels, 0, 10) #duplicate for each crop the label [1,2],[3,4] => [1,2],[1,2],[3,4],[3,4] -> 640 labels

                # forward + backward + optimize
                outputs = model(image_batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def tile(self,a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

def main():
    trainset = DataPreprocessing()
    model = DenseNet121()
    train = Train(trainset, model)

if __name__ == '__main__':
    main()
