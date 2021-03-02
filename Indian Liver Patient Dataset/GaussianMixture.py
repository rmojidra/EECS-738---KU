import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Read the data
data = pd.read_excel('Indian Liver Patient Dataset.xlsx')
data = data.to_numpy()
sepal_length = data[:,2]

# Input: no. of classes, no. of iterations
k = 2 
iter = 150

# Calculate Mean and Standard Deviation
count, bins, ignored = plt.hist(sepal_length, 20, density=True)

# Create Pi,mu, and sigma
mu = np.random.randint(5, size=(k,1))
sigma = np.random.randint(5, size=(k,1))
pi = np.random.dirichlet(np.ones(k),size=1).transpose() 

Gamma = np.zeros((len(bins),k))
pdf = np.zeros((len(bins),k))
xGamma = np.zeros((len(bins),k))
xsigma = np.zeros((len(bins),k))

# Creating a Function for calculating gaussian distribution probability
def normal_dist(data, mean, sd):
    prob_density = 1/(sd*np.sqrt(2*np.pi)) * np.exp(-(data.astype(float) - mean)**2/(2*sd**2))
    return prob_density 

# E STEP CALCULATION
def estep(mu,sigma,pi):
    # Calculate PDF 
    for x in range(k):
        pdf[:,x] = normal_dist(bins,mu[x],sigma[x])

    # Calculate pipdf2
    pipdf2 = pdf.copy()
    for x in range(k):
        pipdf2[:,x] = pi[x] * pipdf2[:,x]
    pipdf2 = pipdf2.sum(axis=1)

    # Calculate pipdf
    pipdf = pdf.copy()
    for x in range(k):
        pipdf[:,x] = pi[x] * pdf[:,x]

    # Calculate Gamma
    for x in range(k):
        Gamma[:,x] = pipdf[:,x] / pipdf2[:]

    return Gamma

# M STEP CALCULATION
def mstep(resposibility):
    xGamma = np.zeros((len(bins),k))
    xsigma = np.zeros((len(bins),k))
    # Calculate NK
    NK = Gamma.sum(axis=0)

    # Calculate new mu
    for x in range(k):
        xGamma[:,x] = bins * Gamma[:,x]
    xGamma = xGamma.sum(axis=0)
    munew = xGamma / NK

    # Calculate Sigma
    for x in range(k):
        xsigma[:,x] = Gamma[:,x] * (bins-munew[x])* (bins-munew[x])
    xsigma = xsigma.sum(axis=0)
    sigmanew = np.sqrt(xsigma / NK)

    # Calculate pinew
    pinew = NK / len(bins)

    return munew,sigmanew,pinew
  
for x in range(iter):
    responsibility = estep(mu,sigma,pi)
    mu,sigma,pi = mstep(responsibility)

pdf1 = normal_dist(bins,mu[0],math.sqrt(sigma[0]))
pdf2 = normal_dist(bins,mu[1],math.sqrt(sigma[1]))

print('mu =', mu)
print('sigma =', sigma)
print('pi =', pi)


#Plotting the results
plt.plot(bins,pi[0]*pdf1, color = 'red')
plt.plot(bins,pi[1]*pdf2, color = 'red')
plt.xlabel('Age')
plt.ylabel('Probability Density')
plt.show() 