# --- Assignment 2: Machine Learning Models
#import packages and files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats #used for sem & paired t-test to test function
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind, pearsonr
import seaborn as sns


neuralResponses_S1 = pd.read_csv('FilesForAssignment2_DV/NeuralResponses_S1.txt', sep=",")
categoryLabels = pd.read_csv('FilesForAssignment2_DV/CategoryLabels.txt', sep=",")
categoryVectors = pd.read_csv('FilesForAssignment2_DV/CategoryVectors.txt', sep=",")
neuralResponses_S2 = pd.read_csv('FilesForAssignment2_DV/NeuralResponses_S2.txt', sep=",")

#Averaging data per category
animate_mask = (categoryVectors['Var1'] == 1)
inanimate_mask = (categoryVectors['Var2'] == 1)
animateObjects = neuralResponses_S1[animate_mask]
inanimateObjects = neuralResponses_S1[inanimate_mask]

averageAnimate = animateObjects.mean(axis=1) #axis=1 averages along de columns
averageInanimate = inanimateObjects.mean(axis=1)

# #Question 1A
# x = ['Animate', 'Inanimate']
# y = [averageAnimate.mean(),averageInanimate.mean()]
# error = [stats.sem(averageAnimate), stats.sem(averageInanimate)]
# plt.figure(figsize=(7,7))
# plt.ylabel('Response Amplitude')
# plt.title('Average Response Amplitude')
# plt.bar(x, y, edgecolor='black')
# plt.errorbar(x, y, yerr=error, fmt='none', color="r",capsize=5)
# plt.axhline(0, color='black')
# # plt.show()


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

# plt.figure(figsize=(7, 7))
# plt.ylabel('Response Amplitude')
# plt.xlabel('Voxel')
# plt.title('Animate minus Inanimate')
# plt.xticks(np.arange(0,21,5)) #ensures the x-axis ticks go by increments of 5
# plt.bar(range(1,21), voxelDifference[:20],edgecolor='black')
# plt.axhline(0, color='black')
# # plt.show()

#Question 2A
animateObjects_2 = neuralResponses_S2[animate_mask]
inanimateObjects_2 = neuralResponses_S2[inanimate_mask]

#training set (first 22 images)
train_animate = animateObjects_2.iloc[:22]
train_inanimate = inanimateObjects_2.iloc[:22]
x_train = pd.concat([train_animate, train_inanimate]) #combines inanimate & animate into 1 training set
y_train = [1] * 22 + [-1] * 22 #first 22 images are animate with label 1; last 22 images are inanimate with label -1

#test set (last 22 images)
test_animate = animateObjects_2.iloc[22:]
test_inanimate = inanimateObjects_2.iloc[22:]
x_test = pd.concat([test_animate, test_inanimate]) #combines inanimate & animate into 1 testing set
y_test = [1] * 22 + [-1] * 22

#train SVM model
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

#test the model
prediction = clf.predict(x_test)

print("the predicted outcomes of the SVM are:")
print(prediction)

print("the real answers:")
print(y_test)

#calculate the accuracy of the SVM
accuracy = accuracy_score(y_test, prediction)
print("the accuracy of the SVM is:")
print(accuracy)

#Question 2B
svm_weights = clf.coef_[0, :20] #weights of the first 20 voxels

averageVoxelAnimate_S2 = animateObjects_2.mean(axis=0) #axis=0 to average voxels (not responses)
averageVoxelInanimate_S2 = inanimateObjects_2.mean(axis=0)
voxelDifference_S2 = averageVoxelAnimate_S2-averageVoxelInanimate_S2
voxelDifference_S2_20 = voxelDifference_S2[:20]

# #scatter plot weights vs. responses
# plt.figure(figsize=(7,7))
# plt.xlabel('Weights')
# plt.ylabel('Responses')
# plt.title('Scatter Plot')
# plt.xlim(-0.06, 0.04)           # set axis limits
# plt.xticks(np.arange(-0.06, 0.041, 0.02))  # set ticks with step 0.02
# plt.ylim(-2, 2)                  # set y-axis limits
# plt.yticks(np.arange(-2, 2.1, 0.5))  # set ticks from -2 to 1.5 in steps of 0.5
# plt.scatter(svm_weights, voxelDifference_S2_20, color='none', edgecolor='blue')
# # plt.show()

#Pearson's r
r_matrix = np.corrcoef(svm_weights, voxelDifference_S2_20)
r_value = r_matrix[0, 1]
print("Pearson's r is:")
print(r_value)

#Question 2D
human_mask = (categoryVectors['Var1'] == 1) & (categoryVectors['Var3'] == 1)
inhuman_mask = (categoryVectors['Var1'] == 1) & (categoryVectors['Var4'] == 1)

human_data = neuralResponses_S2[human_mask]
inhuman_data = neuralResponses_S2[inhuman_mask]

human_data = human_data.iloc[:-4] #delete the last 4 rows

# print(human_data)
# print(inhuman_data)

# training set (first 10 images)
train_human = human_data.iloc[:10]
train_inhuman = inhuman_data.iloc[:10]
x_human_train = pd.concat([train_human, train_inhuman])
y_human_train = [1] * 10 + [-1] * 10

#testing set (last 10 images)
test_human = human_data.iloc[10:]
test_inhuman = inhuman_data.iloc[10:]
x_human_test = pd.concat([test_human, test_inhuman])
y_human_test = [1] * 10 + [-1] * 10

#train SVM model
clf_human = svm.SVC(kernel='linear')
clf_human.fit(x_human_train, y_human_train)

#test model
prediction_human = clf_human.predict(x_human_test)

print("the predicted outcomes of the SVM are:")
print(prediction_human)

print("the real answers:")
print(y_human_test)

#calculate accuracy of SVM
accuracy_human = accuracy_score(y_human_test, prediction_human)
print("the accuracy of the SVM is:")
print(accuracy_human)

#calculate weights
svm_weights_human = clf_human.coef_[0, :20] #weights of first 20 voxels
#Average & difference of voxels for S2
averageVoxel_human = human_data.mean(axis=0) #axis=0 to average voxels (not responses)
averageVoxel_inhuman = inhuman_data.mean(axis=0)
voxelDifference_human = averageVoxel_human - averageVoxel_inhuman
voxelDifference_human = voxelDifference_human[:20] #take first 20 rows

# #scatter plot weights vs. responses
# plt.figure(figsize=(7,7))
# plt.xlabel('Weights')
# plt.ylabel('Responses')
# plt.title('Scatter Plot')
# plt.scatter(svm_weights_human, voxelDifference_human, color='none', edgecolor='blue')
# # plt.show()

r_matrix_human = np.corrcoef(svm_weights_human, voxelDifference_human)
r_value_human = r_matrix[0, 1]
print(r_value_human)

#Section 3
corr_matrix = np.corrcoef(neuralResponses_S1)
# print(matrix)

rdm = 1 - corr_matrix #shape: (88, 88)

plt.figure(figsize=(8,7))
plt.imshow(rdm, cmap='Oranges', origin='upper', vmin=0.8, vmax=1.2)

# X-axis: ticks at 10–80 in steps of 10
x_ticks = np.arange(10, 81, 10)
plt.yticks(x_ticks, x_ticks)

# Y-axis: ticks at 20–80 in steps of 20
y_ticks = np.arange(20, 81, 20)
plt.xticks(y_ticks, y_ticks)

plt.title('Representational Dissimilarity Matrix')
plt.colorbar()
plt.show()

#Question 3B

animacy = categoryVectors['Var1'].values     


animacy_col = animacy.reshape(-1, 1)        
# Compute mask: 0 = same animacy, 1 = different animacy
animacy_mask = (animacy_col != animacy_col.T).astype(int)

# plt.figure(figsize=(8,7))
# plt.imshow(animacy_mask, cmap='gray_r')
# # plt.xlabel('Image Number')
# # plt.ylabel('Image Number')
# plt.title('A: RDM')
# plt.colorbar()
# plt.show()


# Step 3: Select RDM entries
within_animacy = rdm[animacy_mask == 0]     # same animacy
between_animacy = rdm[animacy_mask == 1]    # different animacy

np.fill_diagonal(animacy_mask, -1)     # mark diagonal so it can be excluded
np.fill_diagonal(rdm, np.nan)  # remove self-comparisons from RDM

upper_mask_same = np.triu(animacy_mask == 0, k=1)
upper_mask_diff = np.triu(animacy_mask == 1, k=1)

within_animacy = rdm[upper_mask_same]
between_animacy = rdm[upper_mask_diff]

result, p_val = ttest_ind(within_animacy, between_animacy)

# print(result, p_val)

# x = ['Same', 'Different']
# y = [within_animacy.mean(), between_animacy.mean()]
# error = [stats.sem(within_animacy), stats.sem(between_animacy)]
# plt.figure(figsize=(7,7))
# plt.ylim(0, 1.2)
# plt.title('B: Barplot')
# plt.bar(x, y, edgecolor='black')
# plt.errorbar(x, y, yerr=error, fmt='none', color="r",capsize=5)
# plt.axhline(0, color='black')
# # plt.show()

#3C
RDM_table = pd.read_csv('FilesForAssignment2_DV/BehaviourRDM.csv', sep=",")

# Convert to numpy array (optional but convenient)
behav_rdm = RDM_table.values

# Plot behavioural RDM
plt.figure(figsize=(8,7))
plt.imshow(behav_rdm, cmap='Oranges', origin='upper')
plt.xlabel('Image Number')
plt.ylabel('Image Number')
plt.title('RDM behavioural')
plt.colorbar(label='Dissimilarity')
plt.show()

#3Da

upper_mask = np.triu(np.ones_like(rdm, dtype=bool), k=1)

fMRI_all = rdm[upper_mask]
behav_all = behav_rdm[upper_mask]

r_all, p_all = pearsonr(fMRI_all, behav_all)
print("All categories: r =", r_all, "p =", p_all)


corr_matrix = np.corrcoef(fMRI_all, behav_all)

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("all categories")
plt.xlabel("Behavioral RDM")
plt.ylabel("fMRI RDM")
plt.show()



#3Db only animate
animate_mask = categoryVectors['Var1'].values == 1

fMRI_animate_matrix = rdm[np.ix_(animate_mask, animate_mask)]
behav_animate_matrix = behav_rdm[np.ix_(animate_mask, animate_mask)]

upper_mask = np.triu(np.ones_like(fMRI_animate_matrix, dtype=bool), k=1)

fMRI_only_animate = fMRI_animate_matrix[upper_mask]
behav_only_animate = behav_animate_matrix[upper_mask]

corr_matrix = np.corrcoef(fMRI_only_animate, behav_only_animate)

r_all, p_all = pearsonr(fMRI_only_animate, behav_only_animate)
print("Only animate: r =", r_all, "p =", p_all)

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("only animate")
plt.xlabel("Behavioral RDM")
plt.ylabel("fMRI RDM")
plt.show()

#3Dc

inanimate_mask = categoryVectors['Var1'].values == 0

fMRI_inanimate_matrix = rdm[np.ix_(inanimate_mask, inanimate_mask)]
behav_inanimate_matrix = behav_rdm[np.ix_(inanimate_mask, inanimate_mask)]

upper_mask = np.triu(np.ones_like(fMRI_inanimate_matrix, dtype=bool), k=1)

fMRI_only_inanimate = fMRI_inanimate_matrix[upper_mask]
behav_only_inanimate = behav_inanimate_matrix[upper_mask]

r_all, p_all = pearsonr(fMRI_only_inanimate, behav_only_inanimate)
print("Only inanimate: r =", r_all, "p =", p_all)

corr_matrix = np.corrcoef(fMRI_only_inanimate, behav_only_inanimate)

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("only inanimate")
plt.xlabel("Behavioral RDM")
plt.ylabel("fMRI RDM")
plt.show()
