import torch
from torchvision import transforms
import glob
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import os
import PIL 
from PIL import Image
import numpy as np
import shutil
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import seaborn as sns

# Please specify path to this folder below (change ADD_PATH to location of this folder)
PATH = ('ADD_PATH/AMLS_20-21_20164490/')
# ======================================================================================================================
# Task A1
model = torch.load(PATH + 'A1/A1_model.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.eval()
test_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ])


def predict(file_path):
    A1_print = input("Do you want to print predictions for each image in A1 test set? (Y/N)")
    images = glob.glob(file_path + '*.jpg')
    for image in images:
      image_name = image.split('/')[-1]
      image = Image.open(image)
      image_tensor = test_transforms(image).float()
      image_tensor = image_tensor.unsqueeze_(0)
      image_tensor = image_tensor.to(device=device)
      outputs = model(image_tensor)
      _, preds = torch.max(outputs.data.cpu(), 1)
      if A1_print == "Y":
        if int(preds) == 0:
          verdict = print('Predict person in image', image_name, 'is a woman')
        elif int(preds) ==1: 
          verdict = print('Predict person in image', image_name, 'is a man')
        
predict(PATH + 'Datasets/celeba_test/img/')

#Read in the csv data
data = pd.read_csv(PATH + 'Datasets/celeba_test/labels.csv', delim_whitespace=True)
#Create an index
data.set_index('img_name', inplace=True)
#Instead of values 1,-1 I set 0,1 (replace all -1 with 0)
data.replace(-1,0, inplace= True)

filenames = glob.glob(PATH + 'Datasets/celeba_test/img/*jpg')

#Define a class that will load the data when called
class Gender_loader(Dataset):
    def __init__(self, df, img_dir, transform = None):
        self.dataframe = df
        self.img_dir = img_dir
        self.transform = transform
        self.filename = df.index
        self.label = df.gender.values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.filename[idx]))
        label = self.label[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'label': label}

        return sample

training_df = pd.DataFrame()
training_directory = PATH + 'Datasets/celeba_test/img/'
training_df = training_df.append( data[data.index == filenames])

A1_loader = Gender_loader(data, training_directory, transform=test_transforms)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

BATCH_SIZE=10
A1_test_dl = DataLoader(A1_loader, shuffle=False, batch_size=BATCH_SIZE)

confusion_matrix = torch.zeros(data['gender'].nunique(), data['gender'].nunique())
accuracy = 0
with torch.no_grad():
    for i, inputs in enumerate(A1_test_dl):
        image, target = inputs['image'].cuda(), inputs['label'].cuda()
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        accuracy += (preds == target).sum().item()
        for t, p in zip(target.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

accuracy_as_percentage = (accuracy/len(data)) *100
print('Testing accuracy is', accuracy_as_percentage, 'percent')
x = sns.heatmap(confusion_matrix,annot=True)
plt.show
plt.savefig(PATH + 'A1/A1_test_confusion.png')

torch.cuda.empty_cache()

# ======================================================================================================================
# Task A2
model_A2 = A2(args...)
acc_A2_train = model_A2.train(args...)
acc_A2_test = model_A2.test(args...)
Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
model_B1 = B1(args...)
acc_B1_train = model_B1.train(args...)
acc_B1_test = model_B1.test(args...)
Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
model_B2 = B2(args...)
acc_B2_train = model_B2.train(args...)
acc_B2_test = model_B2.test(args...)
Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))