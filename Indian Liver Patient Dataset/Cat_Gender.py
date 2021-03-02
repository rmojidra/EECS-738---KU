import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('Indian Liver Patient Dataset.xlsx')
data = data.to_numpy()
classdata = data[:,1]

# Create variables corresponding to class type in data
a = 0 # for Male
b = 0 # for Female

for x in classdata:
    if x == 'Male':
        a = a+1
    else:
        b = b+1 
    
# Total weight to find probability/Normalization
sum = a + b

# Probaility of getting Male,Female
pa = a/sum #Male
pb = b/sum #Female

# Print the probility of data type
print('Probaility of Male =', pa)
print('Probaility of Female =', pb)

# Plotting categorical Data to visualize
xaxis = ['Male','Female']
yaxis = [pa,pb]
plt.title('Categorical Distribution of Data')
plt.xlabel('Gender Type')
plt.ylabel('Probability')
plt.bar(xaxis,yaxis)
plt.show() 