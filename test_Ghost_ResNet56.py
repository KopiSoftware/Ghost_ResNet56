import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ghost_net import ghost_net
import Ghost_ResNet

batch_size = 500 


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True)
 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#net = ghost_net(width_mult=1.0)
net = Ghost_ResNet.resnet56()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


net.load_state_dict(torch.load("gRes560.weights"))
net.eval()


def test():
  print('test start')
  correct = 0
  total = 0
  with torch.no_grad():
      i=0
      for i,data in enumerate(testloader):
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          print(batch_size*i)
 
  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))
 
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs, 1)
          c = (predicted == labels).squeeze()
          for i in range(4):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1
  cr,to=0,0
  for i in range(10):
      print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
      cr+=class_correct[i]
      to+=class_total[i]
  mAP = cr/to
  print("mAP:", mAP)
  
  
test()
