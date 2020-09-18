import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import Ghost_ResNet_conv1 as Ghost_ResNet
import numpy as np
from PIL import Image
from torch.optim import lr_scheduler
import ctypes
import signal
import sys
import time

player = ctypes.windll.kernel32
plt.rcParams['figure.dpi']=100 
epochs = 10000
batch_size = 128
lr = 0.1
max_iter = 64000


def sigint_handler(signum, frame):
    global is_sigint_up
    is_sigint_up = True
    torch.save(net.state_dict(), 'gRes56_restore.weights')
    print('Catched interrupt signal!')
    player.Beep(1000,1000)
    sys.exit(0)
    
signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)
is_sigint_up = False



class Cutout(object):
    def __init__(self, hole_size):
        self.hole_size = hole_size

    def __call__(self, img):
        return cutout(img, self.hole_size)


def cutout(img, hole_size):
    y = np.random.randint(32)
    x = np.random.randint(32)

    half_size = hole_size // 2

    x1 = np.clip(x - half_size, 0, 32)
    x2 = np.clip(x + half_size, 0, 32)
    y1 = np.clip(y - half_size, 0, 32)
    y2 = np.clip(y + half_size, 0, 32)

    imgnp = np.array(img)

    imgnp[y1:y2, x1:x2] = 0
    img = Image.fromarray(imgnp.astype('uint8')).convert('RGB')
    return img

transform = transforms.Compose(
    [#transforms.RandomCrop(32, padding=4),
     #Cutout(6),
     #transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False)
 
#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       #download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         #shuffle=False)
 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = Ghost_ResNet.resnet56()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[32000,48000], gamma=0.1)
#optimizer = optim.Adam(net.parameters(), lr=lr)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
total_loss = []
epoch_loss = []

def show_loss(plt_loss):
    plt.plot(range(len(plt_loss)),plt_loss)
    plt.ylim((0,max(plt_loss)))
    plt.show()

def get_lr():
    return optimizer.param_groups[0]['lr']

def train(epochs,resume):
    start = 0
    iternum = 0
    #running_loss=0.
    if resume == True:
        net.load_state_dict(torch.load("gRes56_restore.weights"))
        print('Resumed: weights reloaded')
    begin = time.time()
    for epoch in range(start,epochs):
        #print(optimizer.param_groups[0]['lr'])
        for i, data in enumerate(
                torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=False), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss.append(loss.item())
            epoch_loss.append(loss.item())
            print('[%d, %6d, %.4f, %5d] loss: %.6f' %(epoch, iternum,  get_lr(), (i+1)*batch_size, loss.item())) 
            ###save check point
            if (iternum+1)%3000==0:
                show_loss(epoch_loss)
                torch.save(net.state_dict(), 'gRes56_'+str(i)+'.weights')
            if iternum >= max_iter:
                break
            iternum+=1
        show_loss(epoch_loss)
        show_loss(total_loss)
        epoch_loss.clear()
        if iternum >= max_iter:
            break
        torch.save(net.state_dict(), 'gRes56.weights')
        #player.Beep(1000,1000)
    end = time.time()
    print("total time:",(end-begin)/3600)

    
    
if __name__ == "__main__":
    train(epochs, resume=False)
