import sys
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact
################################################################
# Check the system operating system
################################################################

sys_exec = str(sys.executable)  # location of the python.exe file
sys_os = str(sys.platform)  # os of the machine

windows = 'win32'
linux1 = 'linux'
linux2 = 'linux2'
mac = 'darwin'

if (sys_os == linux1) or (sys_os == linux2):
    print('Linux machine')
elif sys_os == windows:
    print('Windows machine')
else:
    print('Mac machine')

################################################################
# Let the experiment be tossing a coin to analsye the fairness
#  of the coin with p := probability that the result is heads
# in a given toss of the coin.
# Goal: Find posterior probability of unknown for the given
# information/ data.

# We need to compute likelihood (numerator).
# Let's assume that the coin is fair (Prior).
# (subjective belief). The denominstor is probabilty of the given
# data which is 1 in this example. It is very hard to compute.
################################################################

################################################################
# list of number of trials performed of 4 different experiments

n = [1, 10, 50, 100]

# list of number of heads observed (number of success) on the
# corresponding experiment

k = [0, 3, 21, 39]

# List constaining values of p (probability of sucess)

x = 0.01
# empty list for filling prob values
p = []

# for loop to create list of values of p (99 values; 98 sub-intervals)

for i in range(99):
    p.append(x*(i+1))

# Compute Likelihood
# empty list for filling posterior values
post = []

for i in range(len(n)):
    post.append([])
    for j in range(len(p)):
        post[i].append(0.5*(fact(n[i])*p[j]**k[i]*(1-p[j])**(n[i]-k[i]))/float(fact(n[i]-k[i])))

# convert list object as numpy array
post = np.asarray(post)

# Sub-plots of posterior of different experiments

fig, axs = plt.subplots(2, 2)
fig.suptitle('Posterior')
axs[0, 0].plot(p, post[0], 'r-o')
axs[0, 1].plot(p, post[1], 'g-o')
axs[1, 0].plot(p, post[2], 'b-o')
axs[1, 1].plot(p, post[3], 'k-o')

for ax in axs.flat:
    ax.set(xlabel=r'$\theta$', ylabel=r'p($\theta$|D)')


plt.show()
