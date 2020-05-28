import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##Train & Test data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307), (1.3081))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform =transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform =transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

##View Dataset

#functions to show an image
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)

#show images
imshow(torchvision.utils.make_grid(images))
#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


##Define CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


##Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)


##Training
total_loss = 0.0
EPOCHS = 1
for j in range(EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(i, "loss ", loss.item(), outputs, labels)

##Save training model
PATH = './MNIST.pth'
torch.save(net.state_dict(), PATH)

##Test network on a tiny subset of test data
dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(images)
_, predicted = torch.max(outputs.data, 1)
print(predicted)
print(labels)

#print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


##Test:
net = Net()
net.load_state_dict(torch.load(PATH))
correct = 0
total = 0
failed_probabilities = []
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, label = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        if (predicted == label).sum().item() != 4:
            print(imshow(torchvision.utils.make_grid(inputs)), predicted, label)
        total += labels.size(0)
        correct += (predicted == label).sum().item()
        #print(total, correct, "\n")

print('Accuracy of the network on MNIST Data: %d %%' % (
    100 * correct / total))
