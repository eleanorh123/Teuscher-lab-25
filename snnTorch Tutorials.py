import snntorch as snn
import torch


#Training parameters
batch_size = 128
data_path = '/Users/eleanorhall/My Drive (halle@catlin.edu)/Upper School/11th Grade/Honors Compsci III/Teuscher-lab-25/mnist_train_100.csv'
num_classes = 10 #this number is specific to the MINST dataset output classes

#torch variables
dtype = torch.float

#1.2 downloads dataset and loads the data into memory

from torchvision import datasets, transforms
transform = transforms.Compose([transforms.Resize((28,28)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0), (1))]) #0,1 is a good default for neural networks
mnist_train = datasets.MNIST(root='./', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./', train=False, download=True, transform=transform)

from snntorch import utils

subset = 10
mnist_train = utils.data_subset(mnist_train, subset=subset) #reduces the data set by factor of subset (defined)
print(f"The size of mnist_train is {len(mnist_train)}")

#above code was tested and works

#1.3 dataloader will serve the memory in batches
from torch.utils.data import DataLoader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

'''
#temporal dynamics
num_steps = 100 #increases cause the proportion of spike to approach original raw value
raw_vector = torch.ones(num_steps)*0.5
rate_coded_vector = torch.bernoulli(raw_vector) #a bernoulli trial has two outcomes: success/failure. ex: coin flip
print(f"Converted vector: {rate_coded_vector}")
print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time")

#above code was tested and works

#from snntorch import spikegen

data = iter(train_loader)
data_it, targets_it = next(data)
spike_data = spikegen.rate(data_it, num_steps=num_steps) #spikegen.rate converts tensor into Poisson spike trains

print(spike_data.size()) #returns a tensor.
'''

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

#spike_data_sample = spike_data[:,0,0]
#print(spike_data_sample.size())

#above code was tested and works
import matplotlib.animation as animation

'''
plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25) #gain is the spiking frequency
spike_data_sample2 = spike_data[:,0,0]
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample2, fig, ax)

writer = animation.FFMpegWriter(fps=30)
anim.save("test_animation2.mp4", writer=writer)

#following averages the spikes over time and reconstructs input images

plt.figure(facecolor="w")

#for the high spiking frequency
plt.subplot(1,2,1) #sets the axes
plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary') #displays the data as an image
plt.axis('off')
plt.title('Gain = 1')

#for the low spiking frequency
plt.subplot(1,2,2)
plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 0.25')

plt.show()

print(f'the corresponding target is: {targets_it}')

###
#converts feature of intensity into a latency coded response
def convert_to_time(data, tau=5, threshold=0.01):
    #tau is the RC time constant of the circuit. higher tau induces slower firing
    #threshold is membrane potential firing threshold. all values below threshold are clipped and assigned to final time step
    spike_time = tau*torch.log(data/(data-threshold))
    return spike_time

raw_input = torch.arange(0.05, 5, 0.05)
#spike_times = convert_to_time(raw_input)

plt.plot(raw_input, spike_times)
plt.xlabel('Input Value')
plt.ylabel('Spike Time (s)')
plt.show()
#smaller the value, later the spike occurs (exponential)

fig = plt.figure(facecolor="w", figsize=(10,5))
ax = fig.add_subplot(1,1,1)
splt.raster(spike_data[:,0].view(num_steps, -1), ax=ax, s=25, c='black')

plt.title('Input Layer')
plt.xlabel('Time Step')
plt.ylabel('Neuron Number')
plt.show()
#my values are super spread

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear=True)
#adding linear=True linearizes the spike times
fig = plt.figure(facecolor="w", figsize=(10,5))
ax = fig.add_subplot(1,1,1)
splt.raster(spike_data[:,0].view(num_steps, -1), ax=ax, s=25, c='black')
plt.title('Input Layer')
plt.xlabel('Time Step')
plt.ylabel('Neuron Number')
plt.show()

#Latency coding
#Has the advantage of sparsity because neurons are constrained to firing a maximum of once over the time course of interest
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, clip=True, normalize=True, linear=True)
fig = plt.figure(facecolor="w", figsize=(10,5))
ax = fig.add_subplot(1,1,1)
splt.raster(spike_data[:,0].view(num_steps, -1), ax=ax, s=25, c='black')
plt.title('Input Layer')
plt.xlabel('Time Step')
plt.ylabel('Neuron Number')
plt.show()

spike_data_sample = spike_data[:,0,0]
print(spike_data_sample.size())
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)

writer = animation.FFMpegWriter(fps=30)
anim.save("test_animation3.mp4", writer=writer)
print(targets_it[0])

#2.4 Data Modulation
data = torch.Tensor([0,1,0,2,8,-20,20,-5,0,1,0]) #fake time series data
plt.plot(data)
plt.title('fake time-series data')
plt.xlabel('Time Step')
plt.ylabel('voltage (mV)')
plt.show()

#shows the times where the difference between data[T] and data[T+1] is greater than or equal to the threshold
#show the negative spikes (off-spikes) by setting off_spike=True in spikegen.delta
spike_data = spikegen.delta(data, threshold=4) #arbitrarily selected threshold
fig=plt.figure(facecolor="w", figsize=(8,1))
ax = fig.add_subplot(1,1,1)
splt.raster(spike_data, ax=ax, c='black')
plt.title('input neuron')
plt.xlabel('Time Step')
plt.yticks([])
plt.xlim(0, len(data))
plt.show()
print(spike_data)
'''

#TUTORIAL 2
import numpy as np
'''
def plot_mem(mem, title=False):
    if title:
        plt.title(title)
    plt.plot(mem)
    plt.xlabel('Time Step')
    plt.ylabel('Membrane Potential')
    plt.xlim([0,50])
    plt.ylim([0,1])
    plt.show()

def plot_step_current_response(cur_in, mem_rec, vline1):
    fig, ax = plt.subplots(2, figsize=(8,6), sharex=True)

    #plot input current
    ax[0].plot(cur_in, c='tab:pink')
    ax[0].set_ylim([0,0.2])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    ax[0].set_title("Lapiques Neuron Model With Step Input")

    #plot membrane potential
    ax[1].plot(mem_rec)
    ax[1].set_ylim([0,0.6])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")

    if vline1:
        ax[1].axvline(x=vline1, ymin=0, ymax=2.2, alpha=0.25, linestyle='dashed', c='black', linewidth=2, zorder=0, clip_on=False)
    plt.xlabel("Time step")

    plt.show()


def plot_current_pulse_response(cur_in, mem_rec, title, vline1=False, vline2=False, ylim_max1=False):
    fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True)

    # Plot input current
    ax[0].plot(cur_in, c="tab:orange")
    if not ylim_max1:
        ax[0].set_ylim([0, 0.2])
    else:
        ax[0].set_ylim([0, ylim_max1])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    ax[0].set_title(title)

    # Plot membrane potential
    ax[1].plot(mem_rec)
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")

    if vline1:
        ax[1].axvline(x=vline1, ymin=0, ymax=2.2, alpha=0.25, linestyle="dashed", c="black", linewidth=2, zorder=0,
                      clip_on=False)
    if vline2:
        ax[1].axvline(x=vline2, ymin=0, ymax=2.2, alpha=0.25, linestyle="dashed", c="black", linewidth=2, zorder=0,
                      clip_on=False)
    plt.xlabel("Time step")

    plt.show()


def compare_plots(cur1, cur2, cur3, mem1, mem2, mem3, vline1, vline2, vline3, vline4, title):
    # Generate Plots
    fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True)

    # Plot input current
    ax[0].plot(cur1)
    ax[0].plot(cur2)
    ax[0].plot(cur3)
    ax[0].set_ylim([0, 0.2])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    ax[0].set_title(title)

    # Plot membrane potential
    ax[1].plot(mem1)
    ax[1].plot(mem2)
    ax[1].plot(mem3)
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")

    ax[1].axvline(x=vline1, ymin=0, ymax=2.2, alpha=0.25, linestyle="dashed", c="black", linewidth=2, zorder=0,
                  clip_on=False)
    ax[1].axvline(x=vline2, ymin=0, ymax=2.2, alpha=0.25, linestyle="dashed", c="black", linewidth=2, zorder=0,
                  clip_on=False)
    ax[1].axvline(x=vline3, ymin=0, ymax=2.2, alpha=0.25, linestyle="dashed", c="black", linewidth=2, zorder=0,
                  clip_on=False)
    ax[1].axvline(x=vline4, ymin=0, ymax=2.2, alpha=0.25, linestyle="dashed", c="black", linewidth=2, zorder=0,
                  clip_on=False)

    plt.xlabel("Time step")

    plt.show()


def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input current
  ax[0].plot(cur, c="tab:orange")
  ax[0].set_ylim([0, ylim_max1])
  ax[0].set_xlim([0, 200])
  ax[0].set_ylabel("Input Current ($I_{in}$)")
  if title:
    ax[0].set_title(title)

  # Plot membrane potential
  ax[1].plot(mem)
  ax[1].set_ylim([0, ylim_max2])
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
  if thr_line:
    ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  plt.xlabel("Time step")

  # Plot output spike using spikeplot
  splt.raster(spk, ax[2], s=400, c="black", marker="|")
  if vline:
    ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.ylabel("Output spikes")
  plt.yticks([])

  plt.show()


def plot_spk_mem_spk(spk_in, mem, spk_rec, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8, 6), sharex=True,
                           gridspec_kw={'height_ratios': [0.4, 1, 0.4]})

    # Plot input current
    splt.raster(spk_in, ax[0], s=400, c="black", marker="|")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)
    plt.yticks([])

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[1].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk_rec, ax[2], s=400, c="black", marker="|")
    plt.ylabel("Output spikes")
    plt.yticks([])

    plt.show()


def plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0):
    # Generate Plots to Compare Reset Mechanisms
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 6), sharex=True,
                           gridspec_kw={'height_ratios': [0.4, 1, 0.4], 'wspace': 0.05})

    # Reset by Subtraction: input spikes
    splt.raster(spk_in, ax[0][0], s=400, c="black", marker="|")
    ax[0][0].set_ylabel("Input Spikes")
    ax[0][0].set_title("Reset by Subtraction")
    ax[0][0].set_yticks([])

    # Reset by Subtraction: membrane potential
    ax[1][0].plot(mem_rec)
    ax[1][0].set_ylim([0, 0.7])
    ax[1][0].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[1][0].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)

    # Reset by Subtraction: output spikes
    splt.raster(spk_rec, ax[2][0], s=400, c="black", marker="|")
    ax[2][0].set_yticks([])
    ax[2][0].set_xlabel("Time step")
    ax[2][0].set_ylabel("Output Spikes")

    # Reset to Zero: input spikes
    splt.raster(spk_in, ax[0][1], s=400, c="black", marker="|")
    ax[0][1].set_title("Reset to Zero")
    ax[0][1].set_yticks([])

    # Reset to Zero: membrane potential
    ax[1][1].plot(mem_rec0)
    ax[1][1].set_ylim([0, 0.7])
    ax[1][1].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    ax[1][1].set_yticks([])
    ax[2][1].set_xlabel("Time step")

    # Reset to Zero: output spikes
    splt.raster(spk_rec0, ax[2][1], s=400, c="black", marker="|")
    ax[2][1].set_yticks([])

    plt.show()

def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,7), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input spikes
  splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
  ax[0].set_ylabel("Input Spikes")
  ax[0].set_title(title)

  # Plot hidden layer spikes
  splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s = 0.05, c="black")
  ax[1].set_ylabel("Hidden Layer")

  # Plot output spikes
  splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
  ax[2].set_ylabel("Output Spikes")
  ax[2].set_ylim([0, 10])

  plt.show()

#lapicque's neuron

def leaky_integrate_neuron(U, time_step = 1e-3, I=0, R=5e-7, C=1e-10):
    tau = R*C
    U = U + (time_step/tau)*(-U+I*R)
    return U

def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step = 1e-3, R=5.1, C=5e-3):
    tau_mem = R*C
    spk = (mem>threshold)
    mem = mem + (time_step/tau_mem)*(-mem+cur*R)-threshold*spk
    return mem, spk

num_steps = 100
U = 0.9
U_trace = []

for step in range(num_steps):
    U_trace.append(U)
    U = leaky_integrate_neuron(U)

plot_mem(U_trace, "Leaky Neuron Model")

time_step = 1e-3
R = 5
C = 1e-3

lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)
mem = torch.ones(1)*0.9 #the membrane potential is passed as an input
cur_in = torch.zeros(num_steps, 1) #each element of Iin is sequentially passed as an input
spk_out = torch.zeros(1) #initialize output spikes

mem_rec = [mem] #a list to store recordings of membrane potential


#pass updated values at each timestep
for step in range(num_steps):
    spk_out, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)

mem_rec = torch.stack(mem_rec)
plot_mem(mem_rec, 'lapicques neuron model without stimulus')

cur_in = torch.cat((torch.zeros(10,1), torch.ones(190,1)*0.1), 0)
#torch.cat() concatenates the given sequence of tensors. opposite of torch.split()
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]

num_steps = 200
for step in range(num_steps):
    spk_out, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)

mem_rec = torch.stack(mem_rec) #crunches list of tensors into one tensor

plot_step_current_response(cur_in, mem_rec, 10)

print(f'the calculated value of input pulse [A] x resistance [Ω] is: {cur_in[11]*lif1.R} V')
print(f'the simulated value of steady-state membrane potential is: {mem_rec[200][0]} V')

#now the same thing, but with the step input clipped at t=30ms

#input turns on at t=10, and off at t=30
cur_in1 = torch.cat((torch.zeros(10,1), torch.ones(20,1)*0.1, torch.zeros(170,1)), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec1 = [mem]

#simulate neuron
for step in range(num_steps):
    spk_out, mem = lif1(cur_in1[step], mem)
    mem_rec1.append(mem)
mem_rec1 = torch.stack(mem_rec1)

plot_current_pulse_response(cur_in1, mem_rec1, "lapicques neuron model with input pulse", vline1=10, vline2=30)

#same thing again, but shorten time requirement. increase amplitude, decrease time window
cur_in2 = torch.cat((torch.zeros(10,1), torch.ones(10,1)*0.111, torch.zeros(180,1)), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec2 = [mem]

for step in range(num_steps):
    spk_out, mem = lif1(cur_in2[step], mem)
    mem_rec2.append(mem)
mem_rec2 = torch.stack(mem_rec2)

plot_current_pulse_response(cur_in2, mem_rec2, "lapicques neuron model with input pulse: x1/2 pulse width", vline1=10, vline2=20)

#same thing again, but even shorten time requirement. increase amplitude, decrease time window
cur_in3 = torch.cat((torch.zeros(10,1), torch.ones(5,1)*0.147, torch.zeros(185,1)), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec3 = [mem]

for step in range(num_steps):
    spk_out, mem = lif1(cur_in3[step], mem)
    mem_rec3.append(mem)
mem_rec3 = torch.stack(mem_rec3)

plot_current_pulse_response(cur_in3, mem_rec3, "lapicques neuron model with input pulse: x1/2 pulse width", vline1=10, vline2=15)

compare_plots(cur_in1, cur_in2, cur_in3, mem_rec1, mem_rec2, mem_rec3, 10, 15, 20, 30, "lapicques neuron model with input pulse and various iterations")

#now do a spike, not a pulse. this is what will be used in the neural network
cur_in4 = torch.cat((torch.zeros(10,1), torch.ones(1,1)*0.5, torch.zeros(189,1)), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec4 = [mem]

for step in range(num_steps):
    spk_out, mem = lif1(cur_in4[step], mem)
    mem_rec4.append(mem)
mem_rec4 = torch.stack(mem_rec4)

plot_current_pulse_response(cur_in4, mem_rec4, "lapicques neuron model with input spike", vline1=10, ylim_max1=0.6)


#now using leaky_integrate_and_fire() from tutorial 1

lif2 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5)
print(f'membrane potential is: {lif2.R} V')

spk_in = spikegen.rate_conv(torch.ones((num_steps,1))*0.4)

#cur_in5 = torch.cat((torch.zeros(10,1), torch.ones(190,1)*0.3), 0)
mem = torch.ones(1)*0.5
spk_out = torch.zeros(1)
mem_rec5 = [mem]
spk_rec = [spk_out]

for step in range(num_steps):
    spk_out, mem = lif2(spk_in[step], mem)
    mem_rec5.append(mem)
    spk_rec.append(spk_out)

mem_rec5 = torch.stack(mem_rec5)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(spk_in, mem_rec5, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, title='lapicque neuron model with step input')

lif3 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5, reset_mechanism='zero')

spk_in = spikegen.rate_conv(torch.ones((num_steps,1))*0.4)

#cur_in5 = torch.cat((torch.zeros(10,1), torch.ones(190,1)*0.3), 0)
mem = torch.ones(1)*0.5
spk_out = torch.zeros(1)
mem_rec0 = [mem]
spk_rec0 = [spk_out]

for step in range(num_steps):
    spk_out, mem = lif3(spk_in[step], mem)
    mem_rec0.append(mem)
    spk_rec0.append(spk_out)

mem_rec0 = torch.stack(mem_rec0)
spk_rec0 = torch.stack(spk_rec0)

plot_reset_comparison(spk_in, mem_rec5, spk_rec, mem_rec0, spk_rec0)

#comment out here

#leaky integrate-and-fire neuron for actual neural network

def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
    spk = (mem>threshold)
    mem = beta*mem + w*x - spk*threshold
    return spk, mem

lif1 = snn.Leaky(beta=0.8)

delta_t = torch.tensor(1e-3)
tau = torch.tensor(5e-3)
beta = torch.exp(-delta_t/tau) #using equation 3, but in future it should be hardcoded

print(f'the decay rate is: {beta:.3f}')

num_steps = 200
w = 0.21
beta = 0.819

cur_in = torch.cat((torch.zeros(10), torch.ones(190)*w), 0)
mem = torch.ones(1)
spk_out = torch.zeros(1)
mem_rec = []
spk_rec = []


for step in range(num_steps):
    spk, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)
    spk_rec.append(spk)

mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max1=0.5, title='snn.Leaky Neuron Model')
'''

#Finally the NN!!
import torch.nn as nn

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

    #mem2_rec = torch.stack(mem2_rec)
    #spk1_rec = torch.stack(spk1_rec)
    #spk2_rec = torch.stack(spk2_rec)

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
'''

loss_val = torch.zeros((1), dtype=dtype, device=device)
for step in range(num_steps):
    loss_val += loss(mem_rec[step], targets)
print(f'Training loss: {loss_val.item():.3f}') #loss is large

print_batch_accuracy(data, targets, train=True)

#clear stored gradients
optimizer.zero_grad()
#calculate gradients
loss_val.backward()
#weight update
optimizer.step()

spk_rec, mem_rec = net(data.view(batch_size, -1))
loss_val = torch.zeros((1), dtype=dtype, device=device)
for step in range(num_steps):
    loss_val += loss(spk_rec[step], targets)
print(f'Training loss: {loss_val.item():.3f}')
print_batch_accuracy(data, targets, train=True)
'''

num_epochs = 2
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






'''
plot_snn_spikes(spk_in, spk1_rec, spk2_rec, "Fully connected spiking neural network")

fig, ax = plt.subplots(facecolor='w', figsize=(12,7))
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
spk2_rec = spk2_rec.squeeze(1).detach().cpu()

import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
anim = splt.spike_count(spk2_rec, fig, ax, labels=labels, animate=True)

writer = animation.FFMpegWriter(fps=30)
anim.save("snn_animation.mp4", writer=writer)

splt.traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
fig = plt.gcf()
fig.set_size_inches(8,6)


class LeakySurrogate(nn.Module):
    def __init__(self, beta, threshold=1):
        super(LeakySurrogate, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.spike_gradient = self.ATan.apply

    def forward(self, input_, mem):
        spk = self.spike_gradient((mem-self.threshold)) #calls Heaviside function (0 for negative arguments, 1 for positive)
        reset = (self.beta*spk*self.threshold).detach() #remove reset from computational graph
        mem = self.beta * mem + input_ - reset
        return spk, mem

@staticmethod
class ATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem):
        spk = (mem > 0).float()
        ctx.save_for_backward(mem)
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors #retrieve membrane potential
        grad = 1 / (1+(np.pi*mem).pow(2)) * grad_output
        return grad

lif1 = snn.Leaky(beta=0.9)

'''








