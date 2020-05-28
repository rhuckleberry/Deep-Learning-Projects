import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##Simulation Data
def binomial_simulation(num, prob):
    """
    Returns a sequence of events of die tossed with probability of heads = prob

    Input:
    num - number of trials
    prob - probability of heads 0 <= prob <= 1

    Ouput: sequence x1, x2, ... xnum where xi in {0,1} based on
           this binomial distribution model
        * 0 --> tails
        * 1 --> heads

    *Note prob = 0.5 is a fair die, prob != 0.5 is not a fair coin
    """
    binomial_trials = []
    for i in range(num):
        rand_val = random.random()
        trial_val = 0
        if rand_val < prob:
            trial_val = 1

        binomial_trials.append(trial_val)

    return binomial_trials

# trial = binomial_simulation(7, 0.5)
# print(trial)


##Get Data
training_data = []
train_label = []
train_prob = []

test_data = []
test_label = []
test_prob = []

TRIALS = 10000
BATCH_SIZE = 4
NUM = 100
FAIR = 0.5
UNFAIR = 0.5 #chosen randomly below
UNFAIR_LOWER = 0.49
UNFAIR_UPPER = 0.51

##TRAIN Data
for i in range(TRIALS):
    batch_data = []
    batch_label = []
    batch_prob = []
    for j in range(BATCH_SIZE):

        rand = random.randint(0,1)
        if rand == 0:
            #fair die
            trial_seq = binomial_simulation(NUM, FAIR)
            batch_data.append(trial_seq)
            batch_label.append(0)
            batch_prob.append(FAIR)
        else: #rand == 1
            #unfair die

            #randomly chooses UNFAIR value
            UNFAIR = 0.5 # gets updated below
            while (UNFAIR > UNFAIR_LOWER and UNFAIR < UNFAIR_UPPER):
                UNFAIR = random.random()


            trial_seq = binomial_simulation(NUM, UNFAIR)
            batch_data.append(trial_seq)
            batch_label.append(1)
            batch_prob.append(UNFAIR)

    batch_data = torch.tensor(batch_data, dtype = torch.float)
    batch_label = torch.tensor(batch_label, dtype = torch.long)

    training_data.append(batch_data)
    train_label.append(batch_label)
    train_prob.append(batch_prob)

##TEST Data
for i in range(TRIALS):
    batch_data = []
    batch_label = []
    batch_prob = []
    for j in range(BATCH_SIZE):

        rand = random.randint(0,1)
        if rand == 0:
            #fair die
            trial_seq = binomial_simulation(NUM, FAIR)
            batch_data.append(trial_seq)
            batch_label.append(0)
            batch_prob.append(FAIR)
        else:
            #unfair die

            #randomly chooses UNFAIR value
            UNFAIR = 0.5 # gets updated below
            while (UNFAIR > UNFAIR_LOWER and UNFAIR < UNFAIR_UPPER):
                UNFAIR = random.random()

            trial_seq = binomial_simulation(NUM, UNFAIR)
            batch_data.append(trial_seq)
            batch_label.append(1)
            batch_prob.append(UNFAIR)

    batch_data = torch.tensor(batch_data, dtype = torch.float)
    batch_label = torch.tensor(batch_label, dtype = torch.long)

    test_data.append(batch_data)
    test_label.append(batch_label)
    test_prob.append(batch_prob)

# print(train_prob, "\n")

##Neural Network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=100, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=25)
        self.fc3 = nn.Linear(in_features=25, out_features=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #print("fc1: ", x)
        x = F.relu(self.fc2(x))
        #print("fc2: ", x)
        x = self.fc3(x)
        #print("fc2: ", x)
        return x

net = Net()

##Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

##Training
running_loss = 0.0
EPOCHS = 1
for j in range(EPOCHS):
    for i, data in enumerate(training_data, 0):
        inputs = data
        labels = train_label[i]

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(i, "loss ", running_loss, outputs, labels)
        running_loss = 0.0

##Saving NN
PATH = './coin_net.pth'
torch.save(net.state_dict(), PATH)

##Check progress:
##Create batch:
# batch_data = []
# batch_label = []
# for j in range(BATCH_SIZE):
#     rand = random.randint(0,1)
#     if rand == 0:
#         #fair die
#         trial_seq = binomial_simulation(NUM, FAIR)
#         batch_data.append(trial_seq)
#         batch_label.append(0)
#     else:
#         #unfair die
#         trial_seq = binomial_simulation(NUM, UNFAIR)
#         batch_data.append(trial_seq)
#         batch_label.append(1)

# batch_data = torch.tensor(batch_data, dtype = torch.float)
# batch_label = torch.tensor(batch_label, dtype = torch.long)

# #Test batch:
# net = Net()
# net.load_state_dict(torch.load(PATH))

# #test on images above
# outputs = net(batch_data)
# _, predicted = torch.max(outputs, 1)

# print(predicted, batch_label)

##Test:
net = Net()
net.load_state_dict(torch.load(PATH))
correct = 0
total = 0
failed_probabilities = []
with torch.no_grad():
    for i, data in enumerate(test_data, 0):
        inputs = data
        label = test_label[i]
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # if predicted != label:
        #     print(outputs, predicted, label, test_prob[i])
        total += labels.size(0)
        correct += (predicted == label).sum().item()
        #print(total, correct, "\n")

print('Accuracy of the network on binomial sequences: %d %%' % (
    100 * correct / total))
