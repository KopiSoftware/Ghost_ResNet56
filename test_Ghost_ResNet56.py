import torch
import torchvision
import torchvision.transforms as transforms
import Ghost_ResNet_conv1 as Ghost_ResNet
import numpy as np
from PIL import Image

batch_size = 500 

class Cutout(object):
    def __init__(self, hole_size):
        # 正方形马赛克的边长，像素为单位
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
    [transforms.RandomCrop(32, padding=4),
     Cutout(6),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#net = ghost_net(width_mult=1.0)
net = Ghost_ResNet.resnet56()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


net.load_state_dict(torch.load("gRes56_155.weights"))
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
 
  print('Accuracy of the network on the 10000 test images: %4f %%' % (
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
      print('Accuracy of %5s : %4f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
      cr+=class_correct[i]
      to+=class_total[i]
  mAP = cr/to
  print("Acc: %4f"%(mAP*100))
  
  
test()
