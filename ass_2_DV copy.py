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
# plt.show()

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
# plt.show()

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
# plt.show()



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
# plt.show()

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
# plt.show()

# """
# Bonus A
# """

# from scipy.stats import ttest_1samp

# # determine the human and inhuman masks based on values Var3 and Var4
# mask_human = (categoryVectors['Var3'] == 1)
# mask_inhuman = (categoryVectors['Var4'] == 1)

# # use mask to select human/inhuman data points
# human_datapoints = neuralResponses_S2[mask_human]
# inhuman_datapoints = neuralResponses_S2[mask_inhuman]

# # combine the data to make training set
# x = np.concatenate([human_datapoints.values, inhuman_datapoints.values], axis=0)

# # create the labels for the training set
# y = np.array([1]*human_datapoints.shape[0] + [-1]*inhuman_datapoints.shape[0])

# # get the total number of training samples
# len_data = x.shape[0]

# results = []

# for i in range(len_data):
    
#     # get all the training samples + labels except for the ith one
#     train_x = np.concatenate([x[:i], x[i+1:]], axis=0)
#     train_y = np.concatenate([y[:i], y[i+1:]])
    
#     # set ith example + label as test set
#     test_x = x[i].reshape(1, -1)
#     test_y = y[i]
    
#     # fit linear svm to the training set
#     svm = svm.SVC(kernel='linear')
#     svm.fit(train_x, train_y)
    
#     # predict label of the test example
#     prediction = svm.predict(test_x)
    
#     # determine whether prediction is correct
#     if prediction == test_y:
#           results.append(1)
#     else:
#           results.append(0)

# mean_decoding_accuracy = sum(results)/len(results)
# print("Average decoding accuracy :", mean_decoding_accuracy)


# t, p = ttest_1samp(results, 0.5)
# print("p-value:", p)

"""
Bonus B
"""

# using a prior mask variable and selected S1 datapoints
animate_S1 = animateObjects.values
inanimate_S1 = inanimateObjects.values

# combine the S1 data to make training set
train_x_S1 = np.concatenate([animate_S1, inanimate_S1], axis=0)
train_y_S1 = np.array([1]*animate_S1.shape[0] + [-1]*inanimate_S1.shape[0])

# use mask to select human/inhuman data points
animate_S2 = neuralResponses_S2[animate_mask].values
inanimate_S2 = neuralResponses_S2[inanimate_mask].values

# combine the S2 data to make test set
x_test_S2 = np.concatenate([animate_S2, inanimate_S2], axis=0)
y_test_S2 = np.array([1]*animate_S2.shape[0] + [-1]*inanimate_S2.shape[0])

"""
Training on S1, testing on S2
"""
# fit svm on S1 data, predict on test data S2
svm_S1_to_S2 = svm.SVC(kernel='linear')
svm_S1_to_S2.fit(train_x_S1,train_y_S1)
pred_S1_to_S2 = svm_S1_to_S2.predict(x_test_S2)

acc_S1_to_S2 = accuracy_score(y_test_S2, pred_S1_to_S2)
print("Decoding accuracy train S1 test S2:", acc_S1_to_S2)

"""
Training on S2, testing on S1
"""
# fit svm on S2 data, predict on test data S1
svm_S2_to_S1 = svm.SVC(kernel='linear')
svm_S2_to_S1.fit(x_test_S2, y_test_S2)
pred_S2_to_S1 = svm_S2_to_S1.predict(train_x_S1)

acc_S2_to_S1 = accuracy_score(train_y_S1, pred_S2_to_S1)
print("Decoding accuracy train S2 test S1:", acc_S2_to_S1)

"""
We improve the performance of the SVM by focussing on discriminative patterns based on specific voxels 
reacting strongly to either animate or inanimate instead of looking at their overall response amplitudes
(as inspired by self check 3A). Therefore, we compute the difference between the animate and inanimate voxels
(as we did previously for exercise 1 and 2) and then take the top 20 most discriminative voxels to train 
the SVM on. Then we test the model on the matching top 20 voxels of S2. 
"""

# get the difference between animate and inanimate avg voxels
average_voxel_animate_S1 = animateObjects.mean(axis=0)
average_voxel_inanimate_S1 = inanimateObjects.mean(axis=0)
voxel_diff_S1 = np.abs(average_voxel_animate_S1 - average_voxel_inanimate_S1)

# get the top 20 most discriminative voxels
top_voxels = np.argsort(voxel_diff_S1)[-20:]

# train model on only these top 20 voxels
train_x_top20 = pd.concat([animateObjects.iloc[:, top_voxels],
                           inanimateObjects.iloc[:, top_voxels]])

# create the labels for the datapoints
train_y_s1 = [1]*animateObjects.shape[0] + [-1]*inanimateObjects.shape[0]

# define the S2 test data with the same voxels
test_x_top20 = neuralResponses_S2.iloc[:, top_voxels] 

# create the test labels for S2 top voxels
y_test_s2 = [1 if x==1 else -1 for x in categoryVectors['Var1']]

# train svm on the top 20 voxels of S1
svm_cross = svm.SVC(kernel='linear')
svm_cross.fit(train_x_top20, train_y_s1)

# test the svm on the top 20 voxels of S2
pred_cross = svm_cross.predict(test_x_top20)

# compute the decoding accuracy 
acc_cross = accuracy_score(y_test_s2, pred_cross)
print("Decoding accuracy train S1 test S2 top 20 voxels:", acc_cross)

