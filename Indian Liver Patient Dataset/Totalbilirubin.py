import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('Indian Liver Patient Dataset.xlsx')
data = data.to_numpy()
sepal_length = data[:,2]
xlabel = 'Total Bilirubin'
Title = 'Probability Distribution of Data'
# bins and histogram
count, bins, ignored = plt.hist(sepal_length,20, density=True)
print(count)
print(bins)

#Calculate mean and std
mean = np.mean(sepal_length)
sd = np.std(sepal_length)
print(mean)
print(sd)
# Creating a Function
def normal_dist(bins, mean, sd):
    #prob_density = (np.pi*sd) * np.exp(-0.5*((data_age-mean)/sd)**2)
    prob_density = 1/(sd*np.sqrt(2*np.pi)) * np.exp(-(bins.astype(float) - mean)**2/(2*sd**2))
    return prob_density 

#Apply function to data.
pdf = normal_dist(bins,mean,sd)

#Plotting the results
plt.xlabel(xlabel)
plt.ylabel('Probability Density')
plt.title(Title)
plt.show() 