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
# Let the experiment be tossing a coin to analsye the fairness of the coin with p := probability that the 
# result is heads in a given toss of the coin. The coin is tossed n times and let k be the number of times 
# the coin is facing heads. Therefore, the random variale X of the experiment follow a Binomial distribution.
# Goal: Find posterior probability of unknown from the given information/ data.

# We need to compute likelihood (numerator) times prior while ignoring the constant term. Assume the prior
# follows a normal distribution with (mean = 0, and variance = 1,0.5 and 0.1). The denominstor is probabilty 
# of the given data which is is very hard to compute therfore this will be ignored.
##########################################################################################################

# define likelihood of theta given data/ conditional probabilty of data given theta
def likelihood(n,k,theta):
    return fact(n)/float(fact(n-k))*(theta)**k * (1-theta)**(n-k)
    
# define normal density without the coefficient 1/square(2pi)*sigma
def exp_gauss(theta,var):
    return np.exp(-(theta-0.5)**2/(2*var**2))
# list of number of trials performed of 4 different experiments
n = [1, 10, 50, 100]

# list of number of heads observed (number of success) on the corresponding experiment
k = [0, 3, 21, 39]

# List constaining values of p (probability of sucess)
x = 0.01

# List constaining values of p (probability of sucess)
var = [1, 0.5, 0.1]

# empty list for filling prob values
theta = []

# for loop to create list of values of p (99 values; 98 sub-intervals)
for i in range(99):
    theta.append(x*(i+1))

# Compute Likelihood times prior
# empty list for filling posterior values
post = []

# double nested 'for loop' inside a 'for loop' to compute posterior probabilities for n = 1,10,50,100 with k = 0,3,21
# and 39, the corresponding number of heads up. Posterior prob stored in a 3D array (3,4,99)
for i in range(len(var)):
    post.append([])
    for j in range(len(n)):
        post[i].append([])
        for m in range(len(theta)):
            post[i][j].append(0.5*likelihood(n[j], k[j], theta[m])*exp_gauss(theta[m],var[i]))

# convert list object as numpy array
post = np.asarray(post)

##########################################################################################################                         
# Plotting figures
##########################################################################################################
# strings that are labels                           
n_k_str = ['n = 1, k = 0, ', 'n = 10, k = 3', 'n = 50, k = 21 ', 'n = 100, k =39 ']
mu_var_str = [r'$\mu = 0.5, \sigma^2 = 1$', r'$\mu = 0.5, \sigma^2 = 0.5$', r'$\mu = 0.5, \sigma^2 = 0.1$']                              

# Sub-plots of posterior of different experiments where the prior is N(0.5,1) 

fig, axs0 = plt.subplots(2, 2)
fig.suptitle('Posterior')

# labels of top panels
for i in range(2):
    axs0[0,i].text(0.7,0.8,n_k_str[i], fontsize = 12,transform = axs0[0,i].transAxes)
    axs0[0,i].text(0.7,0.75,mu_var_str[0], fontsize = 12,transform = axs0[0,i].transAxes)
# plot top panels
axs0[0, 0].plot(theta, post[0,0,:],  'r-o')
axs0[0, 1].plot(theta, post[0,1,:], 'g-o')

# labels of bottom panels
for i in range(2):
    axs0[1,i].text(0.7, 0.8,n_k_str[2+i], fontsize = 12,transform = axs0[1,i].transAxes)
    axs0[1,i].text(0.7, 0.75,mu_var_str[0], fontsize = 12,transform = axs0[1,i].transAxes)

# plot bottom panels                             
axs0[1, 0].plot(theta, post[0,2,:], 'b-o')
axs0[1, 1].plot(theta, post[0,3,:], 'k-o')

# label x and y axis                              
for ax in axs0.flat:
    ax.set(xlabel=r'$\theta$', ylabel=r'Density $\pi(\theta|X = k)$')


plt.savefig(r'Posterior vs theta 0')


# Sub-plots of posterior of different experiments where the prior is N(0.5,0.5) 
fig, axs1 = plt.subplots(2, 2)
fig.suptitle('Posterior')

# labels of top panels
for i in range(2):
    axs1[0,i].text(0.7,0.8,n_k_str[i], fontsize = 12,transform = axs1[0,i].transAxes)
    axs1[0,i].text(0.7,0.75,mu_var_str[1], fontsize = 12,transform = axs1[0,i].transAxes)
# plot top panels
axs1[0, 0].plot(theta, post[1,0,:],  'r-o')
axs1[0, 1].plot(theta, post[1,1,:], 'g-o')

# labels of bottom panels
for i in range(2):
    axs1[1,i].text(0.7, 0.8,n_k_str[2+i], fontsize = 12,transform = axs1[1,i].transAxes)
    axs1[1,i].text(0.7, 0.75,mu_var_str[1], fontsize = 12,transform = axs1[1,i].transAxes)

# plot bottom panels                             
axs1[1, 0].plot(theta, post[1,2,:], 'b-o')
axs1[1, 1].plot(theta, post[1,3,:], 'k-o')

# label x and y axis                              
for ax in axs1.flat:
    ax.set(xlabel=r'$\theta$', ylabel=r'Density $\pi(\theta|X = k)$')

plt.savefig(r'Posterior vs theta 1')                              
                              
                              
# Sub-plots of posterior of different experiments where the prior is N(0.5,0.1) 

fig, axs2 = plt.subplots(2, 2)
fig.suptitle('Posterior')

# labels of top panels
for i in range(2):
    axs2[0,i].text(0.7, 0.8,n_k_str[i], fontsize = 12,transform = axs2[0,i].transAxes)
    axs2[0,i].text(0.7, 0.75,mu_var_str[2], fontsize = 12,transform = axs2[0,i].transAxes)
# plot top panels
axs2[0, 0].plot(theta, post[2,0,:],  'r-o')
axs2[0, 1].plot(theta, post[2,1,:], 'g-o')

# labels of bottom panels
for i in range(2):
    axs2[1,i].text(0.7, 0.8,n_k_str[2+i], fontsize = 12,transform = axs2[1,i].transAxes)
    axs2[1,i].text(0.7, 0.75,mu_var_str[2], fontsize = 12,transform = axs2[1,i].transAxes)

# plot bottom panels                             
axs2[1, 0].plot(theta, post[2,2,:], 'b-o')
axs2[1, 1].plot(theta, post[2,3,:], 'k-o')

# label x and y axis                              
for ax in axs2.flat:
    ax.set(xlabel=r'$\theta$', ylabel=r'Density $\pi(\theta|X = k)$')

plt.savefig(r'Posterior vs theta 2')                              
                              
                              
                                          

plt.show()
