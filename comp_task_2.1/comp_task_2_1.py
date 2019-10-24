import sys
import os
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
# Change font size, ... 
##########################################################################################################
plt.rcParams.update({'font.size': 16})

##########################################################################################################
# Let the experiment be tossing a coin to analsye the fairness of the coin with p := probability that the 
# result is heads in a given toss of the coin. The coin is tossed n times and let k be the number of times 
# the coin is facing heads. Therefore, the random variale X of the experiment follow a Binomial distribution.
# Goal: Find posterior probability of unknown from the given information/ data.

# We need to compute likelihood (numerator) times prior while ignoring the constant term. Assume the prior
# follows a normal distribution with (mean = 0.7 and 0.3, and variance = 1,0.5 and 0.1). The denominator is probabilty 
# of the given data which is is very hard to compute therfore this will be ignored.
##########################################################################################################

# define likelihood of theta given data/ conditional probabilty of data given theta
def likelihood(n,k,theta):
    return fact(n)/float(fact(k)*fact(n-k))*(theta)**k * (1-theta)**(n-k)
    
# define normal density without the coefficient 1/square(2pi)*sigma
def exp_gauss(theta,mean,var):
    return np.exp(-(theta-mean)**2/(2*var**2))

# list of number of trials performed of 4 different experiments
n = [1, 10, 50, 100]

# list of number of heads observed (number of success) on the corresponding experiment
k = [0, 3, 21, 39]

# Initial value of theta/ step size
dx = 0.01

# empty list for filling prob values
theta = []

# for loop to create list of values of theta (99 values; 98 sub-intervals)
for i in range(99):
    theta.append(dx*(i+1))

# List constaining values of variance of prior 
var = [1, 0.5, 0.1]

# List constaining values of mean of prior
mu = [0.7, 0.3]

# Compute Likelihood times prior
# empty lists for filling posterior values; post_1 is the container for posterior with mean of 0.7 and varying variance
# post_2 is the container for posterior with priors with mean of 0.3 and varying variance
post_1, post_2 = [], []

#post_1
# double nested 'for loop' inside a 'for loop' to compute posterior probabilities for n = 1,10,50,100 with k = 0,3,21
# and 39, the corresponding number of heads up. Posterior prob stored in a 3D array (3,4,99)

for i in range(len(var)):
    post_1.append([])
    for j in range(len(n)):
        post_1[i].append([])
        for m in range(len(theta)):
            post_1[i][j].append(0.5*likelihood(n[j], k[j], theta[m])*exp_gauss(theta[m],mu[0],var[i]))


#post_2
# double nested 'for loop' inside a 'for loop' to compute posterior probabilities for n = 1,10,50,100 with k = 0,3,21
# and 39, the corresponding number of heads up. Posterior prob stored in a 3D array (3,4,99)

for i in range(len(var)):
    post_2.append([])
    for j in range(len(n)):
        post_2[i].append([])
        for m in range(len(theta)):
            post_2[i][j].append(0.5*likelihood(n[j], k[j], theta[m])*exp_gauss(theta[m], mu[1],var[i]))

# convert list objects as numpy array
post_1 = np.asarray(post_1)
post_2 = np.asarray(post_2)

##########################################################################################################                         
# Plotting figures (6 subplots, each with 4 panels)
##########################################################################################################

# strings that are labels                           
n_k_str = ['n = 1, k = 0, ', 'n = 10, k = 3', 'n = 50, k = 21 ', 'n = 100, k =39 ']
mu_var_str_1 = [r'$\mu = 0.7, \sigma^2 = 1$', r'$\mu = 0.7, \sigma^2 = 0.5$', r'$\mu = 0.7, \sigma^2 = 0.1$']                              
mu_var_str_2 = [r'$\mu = 0.3, \sigma^2 = 1$', r'$\mu = 0.3, \sigma^2 = 0.5$', r'$\mu = 0.3, \sigma^2 = 0.1$']                              

for i in range(2):
    if i==0:
        post = post_1
        mu_var_str = mu_var_str_1
    else:
        post = post_2
        mu_var_str = mu_var_str_2
    for j in range(3):
        # Sub-plots of posterior of different experiments where the prior is N(mu,var) 
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Posterior')

        # labels of top panels
        for k in range(2):
            axs[0,k].text(0.725,0.95,n_k_str[k], transform = axs[0,k].transAxes)
            axs[0,k].text(0.725,0.89,mu_var_str[j], transform = axs[0,k].transAxes)
            
        # plot top panels
        axs[0, 0].plot(theta, post[j,0,:],  'r-o')
        axs[0, 1].plot(theta, post[j,1,:], 'g-o')

        # labels of bottom panels
        for k in range(2):
            axs[1,k].text(0.725, 0.95, n_k_str[2+k], transform = axs[1,k].transAxes)
            axs[1,k].text(0.725, 0.89,mu_var_str[j], transform = axs[1,k].transAxes)

        # plot bottom panels                             
        axs[1, 0].plot(theta, post[j,2,:], 'b-o')
        axs[1, 1].plot(theta, post[j,3,:], 'k-o')

        # label x and y axis                              
        for ax in axs.flat:
            ax.set(xlabel=r'$\theta$', ylabel=r'$\pi(\theta|X = k)$')

        plt.show()     
