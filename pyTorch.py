import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import cv2
import os
import random
import numpy as np
from os import listdir
from os.path import isfile, join
from torchvision.utils import save_image
import pandas as pd
import pickle
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Conv2d(3, 3, 3) 
        '''
        self.pool = nn.MaxPool2d(32, 32)
        '''
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(21168, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.5)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(x.size()[0], -1) # flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Conv2d(3, 3, 3) 
        self.pool = nn.MaxPool2d(32, 32)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(21168, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.5)  

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1) # flatten layer

        x = self.conv(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
def load_dataset():

    train_data_path = './dataset/train'
    test_data_path = './dataset/test'
    valid_data_path = './dataset/validation'

    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path,transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path,transform=torchvision.transforms.ToTensor())
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_data_path,transform=torchvision.transforms.ToTensor())

    train_iterator = torch.utils.data.DataLoader(train_dataset,batch_size=32,num_workers=0,shuffle=True)
    test_iterator = torch.utils.data.DataLoader(test_dataset,batch_size=32,num_workers=0,shuffle=True)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset,batch_size=32,num_workers=0,shuffle=True)
    
    return train_iterator, test_iterator, valid_iterator
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc   
def evaluate(model, device, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def train(model, device, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        fx = model(x)
        loss = criterion(fx, y)
        
        
        acc = calculate_accuracy(fx, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)   
if __name__ == "__main__":
    '''
    cv2readsave()
    '''
    '''
    train_iterator, test_iterator, valid_iterator = load_dataset()
    '''
    train_iterator, test_iterator, valid_iterator = load_dataset()
                        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()

    optimizer = optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()
    

    ### Training ###
    EPOCHS = 50
    SAVE_DIR = 'models'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'mlp-mnist.pt')

    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, device, valid_iterator, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:05.2f}% |')
    
        

    ### Testing ###
    
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    test_loss, test_acc = evaluate(model, device, valid_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}% |')