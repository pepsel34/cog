"""
This file can be used to run the Bayesian Sampling model of Zhu et al (2020)
This code is largely the code provided by Zhu et al at this link: https://osf.io/mgcxj/
Their code accompanies their paper:
Zhu, J. Q., Sanborn, A. N., & Chater, N. (2020). The Bayesian sampler: Generic Bayesian inference causes incoherence in human probability judgments. Psychological review, 127(5), 719. https://doi.org/10.1037/rev0000190 

In this very slightly modified version, small changes have been made to fit the purpose of a lab assignment in the course "Cognitive Modeling" (INFOMCM) at Utrecht University,
coordinated by Dr. Chris Janssen.
We are grateful for Zhu et al for openly sharing their work.




"""
import numpy
import numpy as np
import scipy.stats as st
from scipy.optimize import fmin, differential_evolution
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#Variables for Engineering 4.1
FIXED_N = 1
FIXED_BETA = 1

#Formula to calculate Bayesian Information Criterion
def bic_formula(n, parameters, mse):
    bic = n * np.log10(mse) + np.log10(n) * (parameters + 1) + n * np.log10(2*np.pi) + n
    return bic

### Comment Chris: This function processes the subject data. Note that you should have the data stored in a specific subdirectory (all_data)
def clean_data():
    """
    Inputs: raw data
    Outputs: <1> subjective estimates est[i,j,k] --> p_data in main
                 i-th participant, j-th query, and k-th repetition
             <2> queryOrder --> the only order we clean data and generate model predictions
    """
    all_data = glob.glob('all_data/*.csv') # data directory
    numPar = len(all_data) # total no. of participants
    print(numPar,' participants were considered!')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    est = np.zeros(shape=(numPar,60,3))
    # est[i,j,k] meaning:
    # probability estimates of i-th participant, j-th query, and k-th repetition of the same query
    neg, land, lor, lg = ' not', ' and', ' or', ' given'
    eventAs = [' cold', ' windy', ' warm']
    eventBs = [' rainy', ' cloudy', ' snowy']
    queryOrder = []
    for A, B in zip(eventAs, eventBs):
        queryOrder.append(A)
        queryOrder.append(B)
        queryOrder.append(neg + A)
        queryOrder.append(neg + B)
        queryOrder.append(A + land + B)
        queryOrder.append(B + land + neg + A) #queryOrder.append(neg + A + land + B)
        queryOrder.append(A + land + neg + B)
        queryOrder.append(neg + A + land + neg + B)
        queryOrder.append(A + lor + B)
        queryOrder.append(B + lor + neg + A) #queryOrder.append(neg + A + lor + B)
        queryOrder.append(A + lor + neg + B)
        queryOrder.append(neg + A + lor + neg + B)
        queryOrder.append(A + lg + B)
        queryOrder.append(neg + A + lg + B)
        queryOrder.append(A + lg + neg + B)
        queryOrder.append(neg + A + lg + neg + B)
        queryOrder.append(B + lg + A)
        queryOrder.append(neg + B + lg + A)
        queryOrder.append(B + lg + neg + A)
        queryOrder.append(neg + B + lg + neg + A)


    for i,fname in enumerate(all_data): # loop through data files
        print('Processing Participant No.%d' % (i+1), fname)
        print('________________________________________________________')

        df = pd.read_csv(fname)  # read data
        for j,q in enumerate(queryOrder): # loop through query
            nowEst = df[df['querydetail']==q]['estimate']
            nowEstValues = nowEst.values/100
            for k in range(3):
                est[i,j,k] = nowEstValues[k]

    # save cleaned dataset
    with open('pEstData.pkl','wb') as f:
        pickle.dump({'data': est, 'query_order': queryOrder}, f)
    return est, queryOrder

##Comment Chris: function to calculate various probabilities
def get_truePr_BS(a,b,c,d):
    truePr = []
    base = a+b+c+d
    truePr.append((a+c)/base)
    truePr.append((a+b)/base)
    truePr.append((b+d)/base)
    truePr.append((c+d)/base)
    truePr.append(a/base)
    truePr.append(b/base)
    truePr.append(c/base)
    truePr.append(d/base)
    truePr.append((a+b+c)/base)
    truePr.append((a+b+d)/base)
    truePr.append((a+c+d)/base)
    truePr.append((b+c+d)/base)
    truePr.append((a/(a+b)))
    truePr.append((b/(a+b)))
    truePr.append((c/(c+d)))
    truePr.append((d/(c+d)))
    truePr.append((a/(a+c)))
    truePr.append((c/(a+c)))
    truePr.append((b/(b+d)))
    truePr.append((d/(b+d)))

    return truePr


### Comment Chris: function that generates the model and calculates MSE. This function is called by the function MSE_BS
def generativeModel_BS(params):
    """
    notice N and N2 are global parameters predetermined
    give a set of free parameters for BS model
     --> output the model predicted distributions for all querys
    """
    global FIXED_N, FIXED_BETA

    a,b,c,d = [0,0],[0,0],[0,0],[0,0]
    a[0],b[0],c[0],d[0],a[1],b[1],c[1],d[1] = params
    beta = FIXED_BETA
    N = FIXED_N
    N2 = N
    MSE = 0

    allpredmeans = np.zeros((40,))


    for iter in range(2):
        sum_of_truePr = a[iter]+b[iter]+c[iter]+d[iter]
        MSE += (sum_of_truePr/100-1)**2/2 # make sure a+b+c+d is close to 100

        truePr = get_truePr_BS(a[iter], b[iter], c[iter], d[iter])

        for i, trueP in enumerate(truePr):
            if i<4 or i>=12: # simple and conditionals
                allpredmeans[i+iter*20] = trueP*N/(N+2*beta)+beta/(N+2*beta) ### Chris: This is unique to BS model: calculation with N and beta is used
            else: # conjunctions and disjunctions
                allpredmeans[i+iter*20] = trueP*N2/(N2+2*beta)+beta/(N2+2*beta)   ### Chris: This is unique to BS model: calculation with N and beta is used

    return allpredmeans, MSE


### Comment Chris: this function is called by the evolutionaary model. The function gets bounds as parameters and then tries to find the best parameters that fall within those bounds to minimize MSE
def MSE_BS(params):
    """
    compute the MSE distance of model prediction <--> actual data
    """
    allpredmeans, MSE = generativeModel_BS(params)
    for i in range(len(allpredmeans)):
        currentdata = testdata[i,:].flatten()
        MSE += np.mean((allpredmeans[i] - currentdata) ** 2)/40  #### Chris: there are 40 unique sentences to test. So, calculate mean over 40 (20 per category)

    return MSE



## Comment Chris: function that goes through all participants and finds the best fit for them
def init_fit_BS():
    """
    initialize model fitting practice for Bayesian sampling model
    """

    global testdata, FIXED_N, FIXED_BETA
    N_values = [1, 2, 5, 10, 50, 100]
    Beta_values = [0.1, 0.5, 1, 2, 5]
    print(np.shape(pData))
    results_plot = {}
    bnds = [(0.0, 100), (0.0, 100), (0.0, 100), (0.0, 100),
            (0.0, 100), (0.0, 100), (0.0, 100), (0.0, 100)]
    totBIC = 0
    for n_val in N_values:
        FIXED_N = n_val
        results_plot[n_val] = []

        for b_val in Beta_values:
            FIXED_BETA = b_val
            total_mse = 0

            for ipar in range(10):  # loop through participants
                minMSE, n_data, BIC = 0, 0, 0
                n_para = 5  # effective number of parameters: [a,b,c]*2, beta/(n+2beta), beta/(n2+2beta)
                testdata = pData[ipar,:,:]
                fit_all_data = differential_evolution(MSE_BS, bounds=bnds,
                                                      popsize=30,
                                                      disp=False, polish=fmin, tol = 1e-5)
                print(fit_all_data.x, fit_all_data.fun)

                minMSE = fit_all_data.fun
                total_mse += minMSE
                n_data = 40*3

                allpredmeans, _ = generativeModel_BS(fit_all_data.x)

            mean_mse = total_mse / 10 #number of participants
            results_plot[n_val].append(mean_mse)
    
            ### Comment Chris: once you are at the relevant part of the assignment, replace the "=0" with a call to your function that calculates the BIC
            BIC = bic_formula(n_data, n_para, minMSE)
            totBIC += BIC

            print('BIC score of Bayesian sampling model:', BIC)

            # model 1,2 = Bayesian sampling (1:one sample size, 2:two sample sizes)
            # model 3   = Sampling/RF
            saved_location = 'fit_results/part_'+str(ipar)+'_model_1_FixedValues.pkl'
            with open(saved_location, 'wb') as f:
                pickle.dump({'fitResults': fit_all_data,
                             'predmean': allpredmeans,
                             'bic':BIC}, f)

    plt.figure(figsize=(10,6))
    for n_val in N_values:
        plt.plot(Beta_values, results_plot[n_val], marker='o', label=f"N={n_val}")

    plt.xlabel('Beta Value')
    plt.ylabel('Mean MSE')
    plt.legend()
    plt.savefig("engineering_4_1.png")
    plt.show()

    print('total BIC:', totBIC)

################################
#          Main Code           #
################################
global pData
pData, queryOrder = clean_data()

init_fit_BS()


## Comment Chris: this code can be modified to access objects that have been stored in a "pickle" object.
# objects = []
# with open('fit_results/part_0_model_1.pkl', 'rb') as f:
#    while True:
#        try:
#            objects.append(pickle.load(f))
#        except EOFError:
#            break
#
# print(objects[0])
# print(objects[0]['bic'])

# total_bic = 0
#
# for i in range(84):
#     filename = f'fit_results/part_{i}_model_1.pkl'
#     try:
#         with open(filename, 'rb') as f:
#             data = pickle.load(f)
#
#             bic_value = data['bic']
#             total_bic += bic_value
#
#     except FileNotFoundError:
#             print(f'File {filename} not found.')

# print("total BIC voor model 1:", total_bic)
    
