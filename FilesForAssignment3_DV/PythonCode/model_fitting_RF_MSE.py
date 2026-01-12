"""
This file can be used to run the Relative Frequency (RF) model of Zhu et al (2020)
This code is largely the code provided by Zhu et al at this link: https://osf.io/mgcxj/
Their code accompanies their paper:
Zhu, J. Q., Sanborn, A. N., & Chater, N. (2020). The Bayesian sampler: Generic Bayesian inference causes incoherence in human probability judgments. Psychological review, 127(5), 719. https://doi.org/10.1037/rev0000190 

In this very slightly modified version, small changes have been made to fit the purpose of a lab assignment in the course "Cognitive Modeling" (INFOMCM) at Utrecht University,
coordinated by Dr. Chris Janssen.
We are grateful for Zhu et al for openly sharing their work.




"""




import numpy as np
import scipy.stats as st
from scipy.optimize import fmin, differential_evolution
import glob
import pandas as pd
import pickle

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
def get_truePr_RF(a,b,c,d):
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



### Comment Chris: function that generates the model and calculates MSE. This function is called by the function MSE_RF
def generativeModel_RF(params):
    a, b, c, d = [0, 0], [0, 0], [0, 0], [0, 0]
    a[0], b[0], c[0], d[0], a[1], b[1], c[1], d[1] = params
    MSE = 0

    allpredmeans = np.zeros((40,))

    for iter in range(2):           ## COmment Chris: this is the crucial part of the code that differs between the two models
        # make sure a+b+c+d is close to 100
        sum_of_truePr = a[iter]+b[iter]+c[iter]+d[iter]
        MSE += (sum_of_truePr/100-1)**2/2

        truePr = get_truePr_RF(a[iter], b[iter], c[iter], d[iter])

        for i, trueP in enumerate(truePr):
            allpredmeans[i+iter*20] = trueP
    return allpredmeans, MSE


### Comment Chris: this function is called by the evolutionaary model. The function gets bounds as parameters and then tries to find the best parameters that fall within those bounds to minimize MSE
def MSE_RF(params):
    """
    compute the MSE distance of model prediction <--> actual data
    """
    allpredmeans, MSE = generativeModel_RF(params)
    for i in range(len(allpredmeans)):
        currentdata = testdata[i,:].flatten()
        MSE += np.mean((allpredmeans[i] - currentdata) ** 2)/40
    return MSE



## Comment Chris: function that goes through all participants and finds the best fit for them
def init_fit_RF():
    """
    initialize model fitting practice for Relative Frequency (RF) model
    """
    global testdata
    totBIC = 0
    print(np.shape(pData))
    bnds = [(0.0, 100), (0.0, 100), (0.0, 100), (0.0, 100),
            (0.0, 100), (0.0, 100), (0.0, 100), (0.0, 100) ]


    for ipar in range(84):  # loop through participants

        minMSE, n_data, BIC = 0, 0, 0
        n_para = 6  # effective number of parameters: [a,b,c]*2
        testdata = pData[ipar,:,:]
        fit_all_data = differential_evolution(MSE_RF, bounds=bnds,
                                              popsize=30,
                                              disp=False, polish=fmin, tol=1e-5)
        print(fit_all_data.x, fit_all_data.fun)

        minMSE = fit_all_data.fun
        n_data = 40*3

        allpredmeans, _ = generativeModel_RF(fit_all_data.x)

        ### Comment Chris: once you are at the relevant part of the assignment, replace the "=0" with a call to your function that calculates the BIC
        BIC = bic_formula(120, n_para, minMSE)
        totBIC += BIC

        print('BIC score of Sampling model:', BIC)
        # model 1,2 = Bayesian sampling (1:one sample size, 2:two sample sizes)
        # model 3   = Sampling/RF
        saved_location = 'fit_results/part_'+str(ipar)+'_model_3.pkl'
        with open(saved_location, 'wb') as f:
            pickle.dump({'fitResults':fit_all_data,
                         'predmean': allpredmeans,
                         'bic':BIC}, f)


################################
#          Main Code           #
################################
global pData
pData, queryOrder = clean_data()

# init_fit_RF() #comment zodat 'ie niet alles opnieuw runt


## Comment Chris: this code can be modified to access objects that have been stored in a "pickle" object.
# objects = []
# with open('fit_results/part_1_model_3.pkl', 'rb') as f: #1 the result for PrEstExp_811
#    while True:
#        try:
#            objects.append(pickle.load(f))
#        except EOFError:
#            break
#
#    print(objects)
#
#    #Brightspace 2a: MSE rounded to 5 decimals
#    MSE_score = round(objects[0]['fitResults'].fun, 5)
#    print(MSE_score)
#
#    #Brightspace 2b: estimate of model for first index of vector b
#    first_index_vector_b = round(objects[0]['fitResults'].x[1], 5)
#    print(first_index_vector_b)
#
#    #Brightspace 2c: estimate of model for sentence A1 and B1 (number 5 in queryorder)
#    sentence_A1_B1 = round(objects[0]['predmean'][4],5)
#    print(sentence_A1_B1)


### Ik was vergeten de totalBIC te printen tijdens het runnen en ik ben te beroerd om 20 min te wachten om init opnieuw te runnen.
total_bic = 0

for i in range(84):
    filename = f'fit_results/part_{i}_model_3.pkl'
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

            bic_value = data['bic']
            total_bic += bic_value

    except FileNotFoundError:
            print(f'File {filename} not found.')

print("total BIC voor model 3:", total_bic)