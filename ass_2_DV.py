# --- Assignment 2: Machine Learning Models
#import packages and files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats #used for sem & paired t-test
neuralResponses_S1 = pd.read_csv('FilesForAssignment2_DV/NeuralResponses_S1.txt', sep=",")
categoryLabels = pd.read_csv('FilesForAssignment2_DV/CategoryLabels.txt', sep=",")
categoryVectors = pd.read_csv('FilesForAssignment2_DV/CategoryVectors.txt', sep=",")

#Averaging data per category
animate = (categoryVectors['Var1'] == 1)
inanimate = (categoryVectors['Var2'] == 1)
animateObjects = neuralResponses_S1[animate]
inanimateObjects = neuralResponses_S1[inanimate]

averageAnimate = animateObjects.mean(axis=1) #axis=1 averages along de columns
averageInanimate = inanimateObjects.mean(axis=1)

#Question 1A
x = ['Animate', 'Inanimate']
y = [averageAnimate.mean(),averageInanimate.mean()]
error = [stats.sem(averageAnimate), stats.sem(averageInanimate)]
plt.figure(figsize=(7,7))
plt.ylabel('Response Amplitude')
plt.title('Average Response Amplitude')
plt.bar(x, y, edgecolor='black')
plt.errorbar(x, y, yerr=error, fmt='none', color="r",capsize=5)
plt.axhline(0, color='black')
# plt.show()


#Question 1B
def paired_ttest(group1, group2):
        difference = group1.values - group2.values
        m = np.mean(difference)
        s = np.std(difference,ddof=1) #ddof=1 for sample instead of population
        n = len(difference)
        t = m/(s/np.sqrt(n))
        df = n - 1
        return t, df

# print(paired_ttest(averageAnimate,averageInanimate))
# print(stats.ttest_rel(averageAnimate, averageInanimate)) #used to see if paired_ttest function is correct

#Question 1C
averageVoxelAnimate = animateObjects.mean()
averageVoxelInanimate = inanimateObjects.mean()
voxelDifference = averageVoxelAnimate-averageVoxelInanimate

plt.figure(figsize=(7, 7))
plt.ylabel('Response Amplitude')
plt.xlabel('Voxel')
plt.title('Animate minus Inanimate')
plt.xticks(np.arange(0,21,5)) #ensures the x-axis ticks go by increments of 5
plt.bar(range(1,21), voxelDifference[:20],edgecolor='black')
plt.axhline(0, color='black')
plt.show()
