import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1000, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x




# input_image_grey = Image.open("test.jpeg")
# input_image = input_image_grey.convert('RGB')
# preprocess = transforms.Compose([transforms.Resize(256),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                                 ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)
# print(input_batch.shape)
# output = model(input_batch)
# print(output[0])

# put into a class getting data ready

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


def main():
    data = DataPreprocessing()
    print(data.image_names)
    image, lbl = data.__getitem__(1)
    print(image.shape)
    trans = transforms.ToPILImage()
    trans(image[9]).show()

if __name__ == '__main__':
    main()
