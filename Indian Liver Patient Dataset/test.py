import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_excel('Indian Liver Patient Dataset.xlsx')
data = data.to_numpy()
sepal_length = data[:,4]
xlabel = 'Alkphos'
Title = 'Gaussian Mixture Model of Data'
# bins and histogram
count, bins, ignored = plt.hist(sepal_length, 20, density=True)
print(count)
print(bins)

#Calculate mean and std
mean1 = 208.0504464
mean2 = 618.9886199
#mean3 = 2.0794
sd1 = math.sqrt(3.34E+03)
sd2 = math.sqrt(1.45E+05)
#sd3 = math.sqrt(0.0615)
pi1 = 0.799176888
pi2 = 0.200823112
#pi3 = 0.2847

# Creating a Function
def normal_dist(bins, mean, sd):
    #prob_density = (np.pi*sd) * np.exp(-0.5*((data_age-mean)/sd)**2)
    prob_density = 1/(sd*np.sqrt(2*np.pi)) * np.exp(-(bins.astype(float) - mean)**2/(2*sd**2))
    return prob_density 

#Apply function to data.
pdf1 = normal_dist(bins,mean1,sd1)
pdf2 = normal_dist(bins,mean2,sd2)
#pdf3 = normal_dist(bins,mean3,sd3)

#Plotting the results
plt.plot(bins,pi1*pdf1, color = 'red')
plt.plot(bins,pi2*pdf2, color = 'black')
#plt.plot(bins,pi3*pdf3, color = 'pink')
plt.xlabel(xlabel)
plt.ylabel('Probability Density')
plt.title(Title)
plt.show() 