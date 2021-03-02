import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read Excel File and save data in numpy array
data = pd.read_excel('iris.xlsx')
data = data.to_numpy()
classdata = data[:,4]

# Create variables corresponding to class type in data
a = 0 # for Iris-setosa
b = 0 # for Iris-versicolor
c = 0 # for Iris-virginica

for x in classdata:
    if x == 'Iris-setosa':
        a = a+1
    elif x == 'Iris-versicolor':
        b = b+1 
    else:
        c = c+1

# Total weight to find probability/Normalization
sum = a + b + c

# Probaility of getting Iris-setosa, Iris-versicolor, Iris-virginica
pa = a/sum #Iris-setosa
pb = b/sum #Iris-versicolor
pc = c/sum #Iris-virginica

# Plotting categorical Data to visualize
xaxis = ['Iris-setosa','Iris-versicolor','Iris-virginica']
yaxis = [pa,pb,pc]
plt.title('Categorical Distribution of Data')
plt.xlabel('Class Type')
plt.ylabel('Probability')
plt.bar(xaxis,yaxis)
plt.show() 