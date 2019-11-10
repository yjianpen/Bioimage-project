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
import PIL
from os.path import isfile, join
from torchvision.utils import save_image
'''
import matplotlib.pyplot as pyplot
'''
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Conv2d(3, 3, 3) 
        
        self.pool = nn.MaxPool2d(32, 32)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(21168, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.5)  

    def forward(self, x):
        print(x.size())
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(x.size()[0], -1) # flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
'''
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        self.conv = nn.Conv2d(3, 32, 3,stride=1,padding=1)
        self.conv2= nn.Conv2d(32, 32, 3,stride=1,padding=1)
        self.conv3= nn.Conv2d(32, 64, 3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(65536, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.5) 
        self.sigmoid=nn.LogSigmoid() 

    def forward(self, x):


        x = (self.pool(F.relu(self.conv(x))))
        '''
        print("size l1",x.size())
        '''
        x =(self.pool(F.relu(self.conv2(x))))
        x = (self.pool(F.relu(self.conv3(x))))
        '''
        x = self.conv(x)
        '''
        '''
        print("size l3",x.size())
        '''
        x = x.view(x.size()[0], -1) # flatten layer
        '''
        print("size l4",x.size())
        '''
        x = (F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.logsigmoid(x)

        return x
def load_dataset():

    train_data_path = './dataset/train'
    test_data_path = './dataset/test'
    valid_data_path = './dataset/validation'
    transform_train = transforms.Compose([
    torchvision.transforms.Resize((256,256)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    torchvision.transforms.ToTensor()
])


    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path,transform=transform_train)
    
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
    '''
    model = Net2().to(device)
    '''
    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=False)
    '''
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    '''
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()
    

    ### Training ###
    EPOCHS = 100
    SAVE_DIR = 'models'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'mlp-mnist.pt')

    best_valid_loss = float('inf')

    if not os.path.exists('./'+SAVE_DIR):
        os.mkdir('./'+SAVE_DIR)
    '''
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
    '''
    val_list=list()
    epoch_list=list()
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, device, valid_iterator, criterion)
        val_list.append(valid_acc)
        epoch_list.append(epoch)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print("| Epoch: %02d | Train Loss: %.3f | Train Acc: %05.2f | Val. Loss: %.3f | Val. Acc: %05.2f |" % (epoch, train_loss, train_acc*100, valid_loss, valid_acc*100))
    '''
    plt=pyplot.plot(epoch_list,val_list)
    pyplot.show()
    '''
        

    ### Testing ###
    
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    test_loss, test_acc = evaluate(model, device, valid_iterator, criterion)

    print(" | Test Loss: %.3f | Test Acc: %05.2f |" % (test_loss, test_acc*100))