#!/usr/bin/env python
# coding: utf-8

# In[4]:


#  AUTHOR: Reeshma Mantena 
#          Vamsi Krishna Muppala 
#          Vandana Priya Muppala 
#          Venkata Hemanth Srivillibhuth
            
#  FILENAME: information retrieval system for images
#  SPECIFICATION: Train the image data using CNN and predict the images category
#  FOR: CS 5364 Information Retrieval Section 001
# 


# In[5]:


#importing required packages
import numpy as np
import os, os.path
import math
from torchsummary import summary
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.image as mpimg


# In[6]:


#Loading dataset
num_epochs = 20
batch_size = 512
image_height = 100
image_width = 100
num_channels = 3

train_dir = "FilePath/Training"
val_dir = "FilePath/Validation"

#Preprocessing data: ToTensor, Resize and Normalize
transforms_data = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       transforms.Resize((image_height, image_width)),
                                       ])

train_dataset = ImageFolder(train_dir, transform=transforms_data) #loading training dataset
validation_dataset = ImageFolder(val_dir, transform=transforms_data) #Loading testing dataset
num_classes = len(train_dataset.classes)

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_validation = DataLoader(validation_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True)

print('Train Dataset Size: ', len(train_dataset))
print('Validation Dataset Size: ', len(validation_dataset))
print('Class Length: ',num_classes)


# In[7]:


#Ploating Traing and Testing data distribution in each category
def plot_category_counts(path,xlabel,ylabel,title):
    categories = []
    counts = []
    for dir in os.listdir(path):
        categories.append(dir)
        counts.append(len(os.listdir(train_dir+"/"+ dir)))
    
    plt.rcParams["figure.figsize"] = (40,20)
    index = np.arange(len(categories))
    plt.bar(index, counts)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(index, categories, fontsize=15, rotation=90)
    plt.title(title, fontsize=30)
    plt.show()

plot_category_counts(train_dir+"/",'Fruit Categories','Category Counts','Fruit Categories Training Distribution')
plot_category_counts(val_dir+"/",'Fruit Categories','Category Counts','Fruit Categories Testing Distribution')


# In[8]:


#Displaying sample of the images in the dataset
fig, ax = plt.subplots(1,4,figsize=(12, 9), dpi=120)
plt.setp(ax, xticks=[], yticks=[])

ax[0].imshow(mpimg.imread(train_dir+'/Apple Braeburn/0_100.jpg'))
ax[1].imshow(mpimg.imread(train_dir+'/Banana/0_100.jpg'))
ax[2].imshow(mpimg.imread(train_dir+'/Avocado/0_100.jpg'))
ax[3].imshow(mpimg.imread(train_dir+'/Apricot/105_100.jpg'))

plt.show()


# In[9]:


#LRScheduler function can gradually decrease the learning rate value dynamically while training. To prevent model overfitting.
class LRScheduler():
    def __init__(self, optimizer, patience=5, min_lr=1e-7, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',patience=self.patience,factor=self.factor,min_lr=self.min_lr,verbose=True)
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
#EarlyStopping function is used for regularization.It stops training when parameter updates no longer begin to yield improves on a validation set.
class EarlyStopping():
    def __init__(self, patience=5, min_delta=0,save_best=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_best=save_best
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            if self.save_best:
                self.save_best_model()
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_best:
                self.save_best_model()
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    def save_best_model(self):
        print(">>> Saving the current model with the best loss value...")
        print("-"*100)
        torch.save(model.state_dict(), 'best_loss_model.pth')


# In[10]:


#CNN model 
class Fruits_CNN(nn.Module):
    def __init__(self):
        super(Fruits_CNN, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=5, stride=1, padding= 'same')
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16*50*50

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding= 'same')
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32*25*25

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding= 'same')
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        # 64*5*5 = 1600 >> it is in_features value for the self.linear1
        
        self.flatten1 = nn.Flatten()

        self.linear1 = nn.Linear(in_features=1600, out_features=512)
        self.dropout1 = nn.Dropout(p=0.25)
        self.linear2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.maxpool3(out)

        out = self.flatten1(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.linear2(out)

        return out


# In[12]:


model = Fruits_CNN() #CNN model
loss_fn = nn.CrossEntropyLoss() #evaluate model using CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam optimizer to optimize the model
lr_scheduler = LRScheduler(optimizer= optimizer,patience=5,min_lr=1e-7, factor=0.5)
early_stopping = EarlyStopping(patience=15, min_delta=0, save_best=True)
#print model summary
print(summary(model, (num_channels, image_height,image_width),batch_size))


# In[13]:


#Check if CUDA is available
CUDA = torch.cuda.is_available()
if CUDA:
    print(f"Using GPU")


# In[ ]:


#Training the dataset and testing the dataset for the model
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

print('Start Training')
print('*'*100)

for epoch in range(num_epochs):
    start_time = datetime.now()

    # TRAINING
    correct = 0
    iterations = 0
    iter_loss = 0.0

    model.train()
    for i, (inputs, labels) in enumerate(dataloader_train):
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        iter_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        iterations += 1

    train_loss.append(iter_loss / iterations)
    train_accuracy.append(100 * correct / len(train_dataset))

    # TESTING
    loss_testing = 0.0
    correct = 0
    iterations = 0

    model.eval()

    for i, (inputs, labels) in enumerate(dataloader_validation):
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss_testing += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

        iterations += 1

    test_loss.append(loss_testing / iterations)
    test_accuracy.append(100 * correct / len(validation_dataset))

    print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}'
          .format(epoch + 1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))

    end_time = datetime.now()
    epoch_time = (end_time - start_time).total_seconds()
    print("-"*100)
    print('Epoch Time : ', math.floor(epoch_time // 60), ':', math.floor(epoch_time % 60))
    print("-"*100)

    lr_scheduler(test_loss[-1])
    early_stopping(test_loss[-1])
    if early_stopping.early_stop:
        print('*** Early stopping ***')
        break
torch.save(model, 'model.pth')
print("*** Training Completed ***")


# In[ ]:


#Finding training and testing dataset accuracy for the model
def list_of_tensors_to_list(tensor_list):
    accuracy = []
    for tensor_item in tensor_list:
        accuracy.append(tensor_item.item())
    return accuracy

train_accuracy = list_of_tensors_to_list(train_accuracy)
test_accuracy = list_of_tensors_to_list(test_accuracy)


# In[54]:


#Ploting the Loss and Accuracy for Training and Testing dataset
fig, ax = plt.subplots(1,2,figsize=(8,4), dpi=120)

# Loss
ax[0].plot(train_loss, label='Training Loss')
ax[0].plot(test_loss, label='Testing Loss')
ax[0].axis(ymin=-0.10, ymax=10)
ax[0].set_title('Loss Plot')
ax[0].legend()

# Accuracy
ax[1].plot(train_accuracy, label='Training Accuracy')
ax[1].plot(test_accuracy, label='Testing Accuracy')
ax[1].axis(ymin=0, ymax=101)
ax[1].set_title('Accuracy Plot')
ax[1].legend()
plt.show()


# In[24]:


#Building a Predictive System
from PIL import Image
import torch
from torch.autograd import Variable

test_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       transforms.Resize((image_height, image_width)),
                                       ])

model=torch.load('model.pth')
model.eval()

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


# In[21]:


#Return random number of images from the dataset according to the input
def get_random_images(num):
    data = ImageFolder(train_dir, transform=transforms_data)
    classes = data.classes
    print('classes',classes)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels


# In[22]:


#Display the images and predicted labels 
to_pil = transforms.ToPILImage()
images, labels = get_random_images(5)
data = ImageFolder(train_dir, transform=transforms_data)
classes = data.classes
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()

