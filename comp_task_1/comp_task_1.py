import sys
import os
import pylab
import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact
##########################################################################################################
# Check the system operating system
##########################################################################################################

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

print('Current working directory: '+os.getcwd())
##########################################################################################################
# Let the experiment be tossing a coin to analsye the fairness of the coin with p := probability that the 
# result is heads in a given toss of the coin. The coin is tossed n times and let k be the number of times 
# the coin is facing heads. Therefore, the random variale of the experiment follow a Binomial distribution.
# Goal: Find posterior probability of unknown from the given information/ data.

# We need to compute likelihood (numerator).
# Let's assume that the coin is fair (Prior).
# (subjective belief). The denominstor is probabilty of the given
# data which is 1 in this example. It is very hard to compute.
##########################################################################################################

##########################################################################################################
# list of number of trials performed of 4 different experiments
n = [1, 10, 50, 100]

# list of number of heads observed (number of success) on the
# corresponding experiment
k = [0, 3, 21, 39]

# List constaining values of p (probability of sucess)
x = 0.01

# empty list for filling prob values
theta = []

# for loop to create list of values of p (99 values; 98 sub-intervals)
for i in range(99):
    theta.append(x*(i+1))

# Compute Likelihood times prior
# empty list for filling posterior values
post = []

# nested 'for loop' inside a 'for loop' to compute posterior probabilities for n = # 1,10,50,100 with k = 0,3,21
# and 39, the corresponding number of heads up. Posterior prob stored in a 2D array (4,99)
for i in range(len(n)):
    post.append([])
    for j in range(len(theta)):
        post[i].append(0.5*(fact(n[i])*theta[j]**k[i]*(1-theta[j])**(n[i]-k[i]))/float(fact(n[i]-k[i])))

# normalise the densities

# list filled with zeros for the total sum of posterior probabilities for each experiment
tot_sum = pylab.zeros(4)
for i in range(len(n)):
    for j in range(len(theta)):
        tot_sum[i]+=post[i][j]

dx = theta[1]-theta[0]

# empty list for filling area of unnormalised densities
coeff = []
# empty list for filling normalised densities
post_n = []
for i in range(len(n)):
    coeff.append(tot_sum[i]*dx)
    post_n.append	(post[i]/coeff[i])
    print(sum(post_n[i]*dx))


# convert list object as numpy array
post = np.asarray(post)
post_n = np.asarray(post_n)


##########################################################################################################                         
# Plotting figures
##########################################################################################################

# Sub-plots of posterior of different experiments

fig, axs = plt.subplots(2, 2)
fig.suptitle('Posterior')
# string that are labels 
n_str = ['n = 1', 'n = 10', 'n = 50', 'n = 100']

# labels of top panels
for i in range(2):
    axs[0,i].text(0.6,0.75,n_str[i], fontsize = 12,transform = axs[0,i].transAxes)

# plot top panels
axs[0, 0].plot(theta, post[0],  'r-o')
axs[0, 1].plot(theta, post[1], 'g-o')

# labels of bottom panels
for i in range(2):
    axs[1,i].text(0.6, 0.75,n_str[2+i], fontsize = 12,transform = axs[1,i].transAxes)


# plot bottom panels
axs[1, 0].plot(theta, post[2], 'b-o')
axs[1, 1].plot(theta, post[3], 'k-o')

# label x and y axis                              
for ax in axs.flat:
    ax.set(xlabel=r'$\theta$', ylabel=r'Density p($\theta$|D)')

plt.savefig(r'Posterior vs theta')

# Sub-plots of normalised posterior of different experiments
fig, axs1 = plt.subplots(2, 2)
fig.suptitle('Normalised Posterior')

# labels of top panels
for i in range(2):
    axs1[0,i].text(0.6,0.75,n_str[i], fontsize = 12,transform = axs1[0,i].transAxes)

# plot top panels
axs1[0, 0].plot(theta, post_n[0],  'r-o')
axs1[0, 1].plot(theta, post_n[1], 'g-o')

# labels of bottom panels
for i in range(2):
    axs1[1,i].text(0.6, 0.75,n_str[2+i], fontsize = 12,transform = axs1[1,i].transAxes)

# plot bottom panels
axs1[1, 0].plot(theta, post_n[2], 'b-o')
axs1[1, 1].plot(theta, post_n[3], 'k-o')

# label x and y axis                              
for ax in axs1.flat:
    ax.set(xlabel=r'$\theta$', ylabel=r'Normalised Density p($\theta$|D)/c')

plt.savefig(r'Normalised Posterior vs theta')
plt.show()
