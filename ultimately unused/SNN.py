import snntorch as snn
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader


#Training parameters
batch_size = 128
data_path = '/Users/eleanorhall/My Drive (halle@catlin.edu)/Upper School/11th Grade/Honors Compsci III/Teuscher-lab-25/mnist_train_100.csv'
num_classes = 10 #this number is specific to the MINST dataset output classes

#torch variables
dtype = torch.float

#downloads dataset and loads the data into memory

transform = transforms.Compose([transforms.Resize((28,28)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0), (1))]) #0,1 is a good default for neural networks
mnist_train = datasets.MNIST(root='./', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./', train=False, download=True, transform=transform)


subset = 10
mnist_train = utils.data_subset(mnist_train, subset=subset) #reduces the data set by factor of subset (defined)
print(f"The size of mnist_train is {len(mnist_train)}")

#dataloader will serve the memory in batches
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.95
num_steps = 25
lif1 = snn.Leaky(beta=0.8)

device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #layer initialization
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        #hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        #outputs
        mem2_rec = []
        spk2_rec = []

    #input spike train
    #dimensions: 200x1x784
    #spk_in = spikegen.rate_conv(torch.rand((200,784))).unsqueeze(1)
    #print(f'Dimensions of spk_in: {spk_in.size()}')

    #running network simulation
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_rec.append(mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

net = Net().to(device)

#compares index of highest spike count with actual target. match = correct prediction
def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f'Train set accuracy for a single minibatch: {acc*100:.2f}%')
    else:
        print(f'Test set accuracy for a single minibatch: {acc*100:.2f}%')

def train_printer(data, targets, epoch, counter, iter_counter, loss_hist, test_loss_hist, test_data, test_targets):
    print(f'Epoch {epoch}, Iteration {iter_counter}')
    print(f'Train Set Loss: {loss_hist[counter]}:.2f')
    print(f'Test Set Loss: {test_loss_hist[counter]}:.2f')
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print('\n')


#handles taking the softmax of the output layer as well as generating a loss at the output
loss = nn.CrossEntropyLoss()
#robust optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

#flatten input data to a vector of size 784
spk_rec, mem_rec = net(data.view(batch_size, -1))
print(mem_rec.size())

num_epochs = 20
loss_hist = []
test_loss_hist = []
counter = 0

for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(spk_rec[step], targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            test_spk, test_mem = net(test_data.view(batch_size, -1))
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_spk[step], test_targets)
            test_loss_hist.append(test_loss.item())

            if counter % 50 == 0:
                train_printer(data, targets, epoch, counter, iter_counter, loss_hist, test_loss_hist, test_data, test_targets)
            counter += 1
            iter_counter += 1

#plotting loss
fig = plt.figure(facecolor='white', figsize=(10,8))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title('loss curves')
plt.legend(['training loss', 'test loss'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

#test accuracy

total = 0
correct = 0

test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        test_spk, _ = net(data.view(data.size(0), -1))
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted==targets).sum().item()

print(f'Total correctly classified test set images: {correct}/{total}')
print(f'Test Set Accuracy: {100*correct/total:.2f}%')

