# Import python packages and modules 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact
import pylab as pyl
from pandas import read_csv

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
# Debugging info
##########################################################################################################

# 26/10/2019 --- changed the code to get 95% HPD credible regions 
# 26/10/2019 --- increased the value of theta from 99 to 999 to get more data and get theta (low,high) for
# multiple posteriors
# 27/10/2019 --- finally realised that the maximum value of the posterior is less than the area under 
# the posterior density times 0.05, need to fix this issue so I can get 95% HPD regions
# 28/10/2019 --- Fixed the 95% HPD issue by calculating area under denisity curve using Trap rule






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
# follows a normal distribution with (mean = 0.7, 0.5 and 0.3 and variance = 1,0.5 and 0.1). The denominator 
# is probabilty  of the given data which is is very hard to compute therfore this will be ignored.
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

# for loop to create list of values of theta (9999 values; 9998 sub-intervals)
for i in range(99):
    theta.append(dx*(i+1))

# List constaining values of variance of prior 
var = [1, 0.5, 0.1]

# List constaining values of mean of prior
mu = [0.7, 0.5, 0.3]

# Compute Likelihood times prior
# empty lists for filling posterior values
# post_1 is the container for posterior with mean of 0.7 and varying variance
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
# calculating areas of each posterior densities using Trapezoidal rule
##########################################################################################################

# container for sum of each posterior densities of post_1, post_2 and post_3
area_1, area_2, area_3 = [], [], []

# step size
dx = (theta[-1]-theta[0])/float(len(theta))

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
                if theta == 0:
                    sum_temp+=post_tmp[i][j][k]/float(2)
                elif theta == len(theta):
                    sum_temp+= post_tmp[i][j][k]/float(2)
                else:
                    sum_temp+=post_tmp[i][j][k]  	                   
                          
            if l == 0:
                area_1[i].append(dx*sum_temp)
            elif l == 1:
                area_2[i].append(dx*sum_temp)
            else:
                area_3[i].append(dx*sum_temp)

##########################################################################################################
# Calculate 95% HPD credible regions for n = 100
##########################################################################################################

##############################################################################################################
# Let the horizontal line that cuts through density be 0.05*A = y. This algorithm loops through each value to 
# find posterior value that is bigger than y-0.05*y and less than y+0.05*y. The low theta and high theta values
# are recorded when the optimal value is found. For low theta, it loop through the index 0 to maxIndex-1 and for 
# high theta, it loops through the index maxIndex+1 to n-1, where maxIndex corresponds to the index with the 
# highest density value.
###############################################################################################################

# convert list objects as numpy array; extract area of posterior (post_1, post_2, post_3) for n = 100 and 
# multiply it by 0.05
area_1, area_2, area_3 = np.asarray(area_1), np.asarray(area_2), np.asarray(area_3)
hund_1, hund_2, hund_3 = 0.05*area_1[:,-1], 0.05*area_2[:,-1], 0.05*area_3[:,-1]
		
# initiialise a container for filling 95% HPD regions for n = 100
low, high = [], []

# initiialise a container for filling 95% HPD posterior values
lowPost, highPost = [], []

# Start algorithm
for l in range(3):
	low.append([])
	high.append([])
	lowPost.append([])
	highPost.append([])
	if l == 0:
		hund = hund_1
		post_tmp = np.asarray(post_1)
		post_tmp = post_tmp[:,-1,:]
		post_tmp.tolist()            

	elif l == 1:
		hund = hund_2
		post_tmp = []
		for tmp in range(len(post_2)):
			post_tmp.append(post_2[tmp][-1])
	else:
		hund = hund_3
		post_tmp = []
		for tmp in range(len(post_3)):
			post_tmp.append(post_2[tmp][-1])
	for j in range(3):
		tmp_post = np.asarray(post_tmp)
		maxIndex = np.argmax(tmp_post[j])
		for i in range(maxIndex):
			if post_tmp[j][i]> hund[j]-hund[j]*0.05 and post_tmp[j][i] < hund[j]+hund[j]*0.05:
				lowPost[l].append(post_tmp[j][i])
				low[l].append(theta[i])
				break
			else:
				continue
                
		for i in range(maxIndex+1,99,1):
            
			if post_tmp[j][i]> hund[j]-hund[j]*0.05 and post_tmp[j][i] < hund[j]+hund[j]*0.05:
				highPost[l].append(post_tmp[j][i])
				high[l].append(theta[i])
				break
			else:
				continue
                

##########################################################################################################
# Export data
##########################################################################################################

# procedure to get corresponding mean and variance values for the table
mu_stack = []
var_stack = []
for i in range(3):
    x = [mu[i]]*3
    y = [var[i]]*3
    mu_stack.extend(x)
    var_stack.extend(y)

# convert relevant data to numpy objects

low_stack, high_stack = np.asarray(low), np.asarray(high)
mu_stack, var_stack = np.asarray(mu_stack), np.asarray(var_stack)
exit()
# Stack low, high, mu_stack and var_stack as column vectors
data = np.vstack((mu_stack, var_stack, low_stack, high_stack))

# transpose stacked data
data = data.T

# save data as a csv file
np.savetxt('HPD_regions.csv', data, delimeter = ',')

# Read csv file using pandas and remove the header
df = read_csv('HPD_regions.csv', header=None)

# Append headers to the columns
df.columns(r'Prior mean $\mu$',r'Prior variance $\sigma^"$','theta_lo','theta_hi')

# Save the data file as csv
df.to_csv('HPD_regions.csv')


exit()

# convert list objects as numpy array
post_1 = np.asarray(post_1)
post_2 = np.asarray(post_2)
post_3 = np.asarray(post_3)

##########################################################################################################                         
# Plotting figures (9 subplots, each with 4 panels)
##########################################################################################################

# strings that are labels                           
n_k_str = ['n = 1, k = 0, ', 'n = 10, k = 3', 'n = 50, k = 21 ', 'n = 100, k = 39 ']

mu_str_1 = [r'$\mu = 0.7, \sigma^2 = 1$', r'$\mu = 0.7, \sigma^2 = 0.5$', r'$\mu = 0.7, \sigma^2 = 0.1$']                              
mu_str_2 = [r'$\mu = 0.5, \sigma^2 = 1$', r'$\mu = 0.5, \sigma^2 = 0.5$', r'$\mu = 0.5, \sigma^2 = 0.1$']                              
mu_str_3 = [r'$\mu = 0.3, \sigma^2 = 1$', r'$\mu = 0.3, \sigma^2 = 0.5$', r'$\mu = 0.3, \sigma^2 = 0.1$']                              

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


##########################################################################################################                         
# Plotting figures (Prior mean values superimposed) (3 subplots, each with 4 panels)
##########################################################################################################

# Strings that are labels and legends of the plots
var_str = [r'$\sigma^2 = 1$', r'$\sigma^2 = 0.5$', r'$\sigma^2 = 0.1$'] 
mu_var_str_1 = [r'$\mu = 0.7$', r'$\mu = 0.7$', r'$\mu = 0.7$']                              
mu_var_str_2 = [r'$\mu = 0.5$', r'$\mu = 0.5$', r'$\mu = 0.5$']                              
mu_var_str_3 = [r'$\mu = 0.3$', r'$\mu = 0.3$', r'$\mu = 0.3$']

# string that contains marker for posterior densities      
dens_marker = ['r-o', 'g*-', 'bs-' ]

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

    # Sub-plots of posterior of different experiments where the prior is N(mu,var)
    # each panel contains three density curves with fixed prior mean and different prior variance
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Posterior')  

    # labels of top panels
    for k in range(2):
        axs[0,k].text(0.735,0.6,n_k_str[k], transform = axs[0,k].transAxes)
        axs[0,k].text(0.735,0.53,mu_var_str[j], transform = axs[0,k].transAxes)
    
    # plot top panels
    for j in range(3):
        axs[0,0].plot(theta, post[j,0,:], dens_marker[j], label = var_str[j])
        axs[0,1].plot(theta, post[j,1,:], dens_marker[j], label = var_str[j])
    axs[0,0].legend(loc = 'best')
    axs[0,1].legend(loc = 'best')
    # plot bottom panels
    for j in range(3):
        axs[1, 0].plot(theta, post[j,2,:], dens_marker[j], label = var_str[j])
        axs[1, 1].plot(theta, post[j,3,:], dens_marker[j], label = var_str[j])
    axs[1,0].legend(loc = 'best')
    axs[1,1].legend(loc = 'best')
    # labels of bottom panels
    for k in range(2):
        axs[1,k].text(0.735, 0.6, n_k_str[2+k], transform = axs[1,k].transAxes)
        axs[1,k].text(0.735, 0.53,mu_var_str[j], transform = axs[1,k].transAxes)

    # label x and y axis                              
    for ax in axs.flat:
        ax.set(xlabel=r'$\theta$', ylabel=r'$\pi(\theta|X = k)$')
    plt.show()