import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('Indian Liver Patient Dataset.xlsx')
data = data.to_numpy()
classdata = data[:,10]

# Create variables corresponding to class type in data
a = 0 # for SET 1
b = 0 # for SET 2

for x in classdata:
    if x == 1:
        a = a+1
    else:
        b = b+1 
    
# Total weight to find probability/Normalization
sum = a + b

# Probaility of getting Male,Female
pa = a/sum #SET 1
pb = b/sum #SET 2

print('Probability of SET 1 =', pa)
print('Probability of SET 2 =', pb)

# Plotting categorical Data to visualize
xaxis = ['SET 1','SET 2']
yaxis = [pa,pb]
plt.title('Categorical Distribution of Data')
plt.xlabel('SET Type')
plt.ylabel('Probability')
plt.bar(xaxis,yaxis)
plt.show() 