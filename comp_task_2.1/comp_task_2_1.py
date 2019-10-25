import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact
from pylab import *
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
# follows a normal distribution with (mean = 0.7, 0.5 and 0.3 and variance = 1,0.5 and 0.1). The denominator is probabilty 
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
mu = [0.7, 0.5, 0.3]

# Compute Likelihood times prior
# empty lists for filling posterior values; post_1 is the container for posterior with mean of 0.7 and varying variance
# post_2 is the container for posterior with priors with mean of 0.5 and varying variance
# post_3 is the container for posterior with priors with mean of 0.3 and varying variance
post_1, post_2, post_3 = [], [], []

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

#post_3
# double nested 'for loop' inside a 'for loop' to compute posterior probabilities for n = 1,10,50,100 with k = 0,3,21
# and 39, the corresponding number of heads up. Posterior prob stored in a 3D array (3,4,99)

for i in range(len(var)):
    post_3.append([])
    for j in range(len(n)):
        post_3[i].append([])
        for m in range(len(theta)):
            post_3[i][j].append(0.5*likelihood(n[j], k[j], theta[m])*exp_gauss(theta[m], mu[2],var[i]))

##########################################################################################################
# calculating areas of each posterior densities
##########################################################################################################

# container for sum of each posterior densities of post_1, post_! and post_3
area_1, area_2, area_3 = [], [], []

# begin calculation
for l in range(3):
    if l == 0:
        post_tmp = post_1
    elif l == 1:
        post_tmp = post_2
    else:
        post_tmp = post_3
    for i in range(len(var)):
        if l == 0:
            area_1.append([])
        elif l == 1:
            area_2.append([])
        else:
            area_3.append([])
        for j in range(len(n)):
            sum_temp = 0
            for k in range(len(theta)):
                sum_temp+=post_1[i][j][k]            
            if l == 0:
                area_1[i].append(sum_temp)
            elif l == 1:
                area_2[i].append(sum_temp)
            else:
                area_3[i].append(sum_temp)

# convert list objects as numpy array; extract area of posterior for n = 100 and multiply by 0.05
area_1, area_2, area_3 = np.asarray(area_1), np.asarray(area_2), np.asarray(area_3)
hund_1, hund_2, hund_3 = 0.05*area_1[:,-1], 0.05*area_2[:,-1], 0.05*area_3[:,-1]
print(hund_1)
print(post_1[0][-1])
## convert back to list: area_1.tolist()
hund = hund_1[0]
count = 0

for i in range(len(theta)+1):
    while i < 99:
        if post_1[0][-1][i]> hund-0.02 and post_1[0][-1][i] < hund+0.02:
            low = post_1[0][-1][i]
            lowIndex = i
            print(i)
        else:
            break

        
    while i < 99:
        if post_1[0][-1][i]> hund-0.02 and post_1[0][-1][i] < hund+0.02:
            high = post_1[0][-1][i]
            highIndex = i
            break

            

print(low,high,lowIndex, highIndex)
print(post_1[0][-1][98])
exit()


# convert list objects as numpy array
post_1 = np.asarray(post_1)
post_2 = np.asarray(post_2)
post_3 = np.asarray(post_3)

##########################################################################################################                         
# Plotting figures (9 subplots, each with 4 panels)
##########################################################################################################

# strings that are labels                           
n_k_str = ['n = 1, k = 0, ', 'n = 10, k = 3', 'n = 50, k = 21 ', 'n = 100, k =39 ']
mu_var_str_1 = [r'$\mu = 0.7, \sigma^2 = 1$', r'$\mu = 0.7, \sigma^2 = 0.5$', r'$\mu = 0.7, \sigma^2 = 0.1$']                              
mu_var_str_2 = [r'$\mu = 0.5, \sigma^2 = 1$', r'$\mu = 0.5, \sigma^2 = 0.5$', r'$\mu = 0.5, \sigma^2 = 0.1$']                              
mu_var_str_3 = [r'$\mu = 0.3, \sigma^2 = 1$', r'$\mu = 0.3, \sigma^2 = 0.5$', r'$\mu = 0.3, \sigma^2 = 0.1$']                              

for i in range(3):
    if i==0:
        post = post_1
        mu_var_str = mu_var_str_1
    elif i==1:
        post = post_2
        mu_var_str = mu_var_str_2
    else:
        post = post_3
        mu_var_str = mu_var_str_3
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
