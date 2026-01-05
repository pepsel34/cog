# --- Assignment 3: Probabilistic Models

#bayesFunction with 2 hypotheses (output is posterior P(H|D))
def bayesFunction(h, likelihood, not_likelihood):
    not_h = 1 - h

    posterior = (h * likelihood)/(h*likelihood + not_h * not_likelihood)
    return posterior

#self-test 1.2
# print(bayesFunction(0.9,0.9,0.3))

#Bayes rule with multiple hypotheses, where the arguments are vectors
#priors[0] = h; all others are "not_h"
def bayesFunctionMultipleHypotheses(priors, likelihoods):
    data = 0
    #calculate data = priors[1]*likelihoods[1] + priors[2]*likelihoods[2] + ...
    for i in range(len(priors)):
        data += priors[i]*likelihoods[i]

    h1_prior = priors[0]
    h1_likelihood = likelihoods[0]

    posterior = (h1_prior*h1_likelihood)/data
    return posterior

# print(bayesFunctionMultipleHypotheses([0.4,0.3,0.3],[0.99,0.9,0.2]))

# #Bayes factor = posteriors / priors.
# def bayesFactor(posteriors, priors):
#     bayes_factors = []
#
#     #calculate BF for h1 vs rest
#     prior_odds_1_vs_rest = priors[0]/(1 - priors[0]) #sum of priors = 1
#     posterior_odds_1_vs_rest = posteriors[0]/(1 - posteriors[0])
#
#     bf_1_vs_rest = posterior_odds_1_vs_rest/prior_odds_1_vs_rest
#     bayes_factors.append(bf_1_vs_rest)
#
#     #calculate BF for h1 vs h2, h1 vs h3, etc.
#     for i in range(1, len(priors)): #skip priors[0] because that is h1
#         prior_odds_1_vs_i = priors[0] / priors[i]
#         posterior_odds_1_vs_i = posteriors[0] / priors[i]
#
#         bf_1_vs_i = posterior_odds_1_vs_i/prior_odds_1_vs_i
#         bayes_factors.append(bf_1_vs_i)
#
#
#     return bayes_factors
#
# print(bayesFactor([0.9,0.05,0.05], [0.2,0.6,0.2]))

def bayesFactor(posteriors, priors):
    #calculate BF for h1 vs rest
    prior_odds_1_vs_rest = priors[0]/(1 - priors[0]) #sum of priors = 1
    posterior_odds_1_vs_rest = posteriors[0]/(1 - posteriors[0])

    bf_1_vs_rest = posterior_odds_1_vs_rest/prior_odds_1_vs_rest
    print(f"BF 1 vs not 1: {bf_1_vs_rest}")

    #calculate BF for h1 vs h2, h1 vs h3, etc.
    for i in range(1, len(priors)): #skip priors[0] because that is h1
        prior_odds_1_vs_i = priors[0] / priors[i]
        posterior_odds_1_vs_i = posteriors[0] / posteriors[i]

        bf_1_vs_i = posterior_odds_1_vs_i/prior_odds_1_vs_i
        print(f"BF 1 vs {i+1}: {bf_1_vs_i}")

# bayesFactor([0.9,0.05,0.05], [0.2,0.6,0.2])
# bayesFactor([0.85,0.05,0.1],[0.2,0.6,0.2])

#Brighspace question 1A - posterior odds
posterior_1A = bayesFunction(0.5,0.531,0.52)
print(posterior_1A) #0.5052331113225499

#Brightspace question 1B - BayesFactor (BF see vs not_see)
bayesFactor([posterior_1A],[0.5]) #BF 1 vs not 1: 1.0211538461538459

#Brightspace question 1C - posterior odds for skeptic researcher
posterior_1C = bayesFunction(0.001, 0.531, 0.52)
posterior_odds_1C = posterior_1C/(1-posterior_1C)
print(posterior_odds_1C) #0.0010211322452794269

#Brightspace question 1D
posterior_ex2 = bayesFunction(posterior_1A, 0.471, 0.520)
print(f"posterior for experiment 2: {posterior_ex2}") #0.48050051777037883
posterior_ex3 = bayesFunction(posterior_ex2,0.491, 0.65)
print(f"posterior for experiment 3: {posterior_ex3}") #0.4113068034046174
posterior_ex4 = bayesFunction(posterior_ex3, 0.505, 0.70)
print(f"posterior for experiment 4: {posterior_ex4}") #0.3351267396958185

#Brighspace question 1E
bayesFactor([posterior_ex4],[0.5]) #last posterior and first prior because all priors are based on the previous posteriors