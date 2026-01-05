import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Importing the files
"""

# from FilesForAssignment3_PS.PythonCode import model_fitting_BS_MSE, model_fitting_RF_MSE


"""
Question 1.1
"""

def bayesRule(P_H, P_DH, P_D_not_H):
    # calculate P_not_H to multiply with P_D_not_H (and add P(H ^ D)) to calculate P_D

    P_not_H = 1 - P_H

    # Calculate P_D

    P_D = P_H * P_DH + P_not_H * P_D_not_H

    # Apply Bayes' Rule

    P_HD = (P_H  * P_DH)/P_D
    
    return round(P_HD, 3)

"""
Test bayesRule function
"""

# testDic = {'A': [0.1, 0.9, 0.3, 0.25], 'B': [0.9, 0.9, 0.3, 0.96], 'C': [0.9, 0.3, 0.9, 0.75],
#             'D': [0.001, 0.99, 0.02, 0.047], 'E': [0.3, 0.5, 0.5, 0.3]}

# for value in testDic.values():
#     P_HD = bayesRule(value[0], value[1], value[2])
#     print(P_HD)
#     if P_HD == value[3]:
#         print(P_HD)

"""
Question 1.2
"""

def bayesFunctionMultipleHypotheses(priors, likelihoods):

    # Calculate P_D by multiplying items from priors and likelihoods and add them all together

    P_D = 0
    for prior, likelihood in zip(priors, likelihoods):
        P_D += (prior * likelihood)

    # Apply Bayes' Rule

    P_H1D = (priors[0] * likelihoods[0])/P_D
    
    return round(P_H1D, 3)

# """
# Test bayesFunctionMultipleHypotheses function
# """

# testDic = {'F': [[0.4, 0.3, 0.3], [0.99, 0.9, 0.2]], 'G': [[0.4, 0.3, 0.3], [0.9, 0.9, 0.2]]}

# for value in testDic.values():
#     P_HD = bayesFunctionMultipleHypotheses(value[0], value[1])
#     print(P_HD)

"""
Question 1.3
"""

def bayesFactor(posteriors, priors):

    post_odds = []

    # Calculate BF 1 vs not 1 (see example teacher)

    BF1_vs_not1 = (posteriors[0] / (1 - posteriors[0]))/(priors[0] / (1 - priors[0]))

    post_odds.append(BF1_vs_not1)

    # Onderste deel uitgecomment om vraag 1B te beantwoorden

    # N = len(posteriors) 

    # # Calculate all the factors BF except BF 1 vs not 1
    # for i in range(1, N):
    #     BF_i = (posteriors[0] / (posteriors[i]))/(priors[0] / (priors[i]))
    #     post_odds.append(BF_i)

    return post_odds

"""
Test bayesFactor function
"""

# testDic = {'A': [[0.9,0.05,0.05], [0.2,0.6,0.2]], 'B': [[0.85,0.05,0.1], [0.2,0.6,0.2]]}

# for value in testDic.values():
#     post_odds = bayesFactor(value[0], value[1])
#     print(post_odds)


"""
Answering question 1
"""

# Question 1A
# P(H) = 0.5, P_DH = 0.531, P_D_not_H = 0.52
posterior_future = bayesRule(0.5, 0.531, 0.52)
# print(posterior_future)

# Question 1B
BF1_vs_not_1 = bayesFactor([posterior_future], [0.5])
# print(BF1_vs_not_1)

# Question 1C
# P(H) = 0.001, P_DH = 0.531, P_D_not_H = 0.52
posterior_future = bayesRule(0.001, 0.531, 0.52)
# print(posterior_future)

# Question 1D
# P(H) = 0.5, P_DH = 0.531, P_D_not_H = 0.52

trials = [[0.471, 0.520], [0.491, 0.65], [0.505, 0.70]]

for i in range(0, len(trials)):
    posterior_future = bayesRule(0.5, trials[i][0], trials[i][1])
    print(posterior_future)

# Question 1E
    
