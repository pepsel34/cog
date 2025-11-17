### 
### This code is developed by Christian P. Janssen of Utrecht University
### It is intended for students from the Master's course Cognitive Modeling
### Large parts are based on the following research papers:
### Janssen, C. P., & Brumby, D. P. (2010). Strategic adaptation to performance objectives in a dual‐task setting. Cognitive science, 34(8), 1548-1560. https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01124.x
### Janssen, C. P., Brumby, D. P., & Garnett, R. (2012). Natural break points: The influence of priorities and cognitive and motor cues on dual-task interleaving. Journal of Cognitive Engineering and Decision Making, 6(1), 5-29. https://journals.sagepub.com/doi/abs/10.1177/1555343411432339
###
### If you want to use this code for anything outside of its intended purposes (training of AI students at Utrecht University), please contact the author:
### c.p.janssen@uu.nl



### 
### import packages
###

import numpy
import matplotlib.pyplot as plt


###
###
### Global parameters. These can be called within functions to change (Python: make sure to call GLOBAL)
###
###


###
### Car / driving related parameters
###
steeringUpdateTime = 250    #in ms ## How long does one steering update take? (250 ms consistent with Salvucci 2005 Cognitive Science)
timeStepPerDriftUpdate = 50 ### msec: what is the time interval between two updates of lateral position?
startingPositionInLane = 0.27 			#assume that car starts already slightly away from lane centre (in meters) (cf. Janssen & Brumby, 2010)


#parameters for deviations in car drift due the simulator environment: See Janssen & Brumby (2010) page 1555
gaussDeviateMean = 0
gaussDeviateSD = 0.13 ##in meter/sec



### The car is controlled using a steering wheel that has a maximum angle. Therefore, there is also a maximum to the lateral velocity coming from a steering update
maxLateralVelocity = 1.7	# in m/s: maximum lateral velocity: what is the maximum that you can steer?
minLateralVelocity = -1* maxLateralVelocity

startvelocity = 0 	#a global parameter used to store the lateral velocity of the car


###
### Switch related parameters
###
retrievalTimeWord = 200   #ms. ## How long does it take to think of the next word when interleaving after a word (time not spent driving, but drifting)
retrievalTimeSentence = 300 #ms. ## how long does it take to retrieve a sentence from memory (time not spent driving, but drifting)



###
### parameters for typing task
###
timePerWord = 0  ### ms ## How much time does one word take
wordsPerMinuteMean = 39.33   # parameters that control typing speed: when typing two fingers, on average you type this many words per minute. From Jiang et al. (2020; CHI)
wordsPerMinuteSD = 10.3 ## this si standard deviation (Jiang et al, 2020)


## Function to reset all parameters. Call this function at the start of each simulated trial. Make sure to reset GLOBAL parameters.
def resetParameters():
    global timePerWord
    global retrievalTimeWord
    global retrievalTimeSentence 
    global steeringUpdateTime 
    global startingPositionInLane 
    global gaussDeviateMean
    global gaussDeviateSD 
    global gaussDriveNoiseMean 
    global gaussDriveNoiseSD 
    global timeStepPerDriftUpdate 
    global maxLateralVelocity 
    global minLateralVelocity 
    global startvelocity
    global wordsPerMinuteMean
    global wordsPerMinuteSD
    
    timePerWord = 0  ### ms

    retrievalTimeWord = 200   #ms
    retrievalTimeSentence = 300 #ms
	
    steeringUpdateTime = 250    #in ms
    startingPositionInLane = 0.27 			#assume that car starts already away from lane centre (in meters)
	

    gaussDeviateMean = 0
    gaussDeviateSD = 0.13 ##in meter/sec
    gaussDriveNoiseMean = 0
    gaussDriveNoiseSD = 0.1	#in meter/sec
    timeStepPerDriftUpdate = 50 ### msec: what is the time interval between two updates of lateral position?
    maxLateralVelocity = 1.7	# in m/s: maximum lateral velocity: what is the maximum that you can steer?
    minLateralVelocity = -1* maxLateralVelocity
    startvelocity = 0 	#a global parameter used to store the lateral velocity of the car
    wordsPerMinuteMean = 39.33
    wordsPerMinuteSD = 10.3

	



##calculates if the car is not accelerating (m/s) more than it should (maxLateralVelocity) or less than it should (minLateralVelocity)  (done for a vector of numbers)
def velocityCheckForVectors(velocityVectors):
    global maxLateralVelocity
    global minLateralVelocity

    velocityVectorsLoc = velocityVectors

    if (type(velocityVectorsLoc) is list):
            ### this can be done faster with for example numpy functions
        velocityVectorsLoc = velocityVectors
        for i in range(len(velocityVectorsLoc)):
            if(velocityVectorsLoc[i]>1.7):
                velocityVectorsLoc[i] = 1.7
            elif (velocityVectorsLoc[i] < -1.7):
                velocityVectorsLoc[i] = -1.7
    else:
        if(velocityVectorsLoc > 1.7):
            velocityVectorsLoc = 1.7
        elif (velocityVectorsLoc < -1.7):
            velocityVectorsLoc = -1.7

    return velocityVectorsLoc  ### in m/s
	




## Function to determine lateral velocity (controlled with steering wheel) based on where car is currently positioned. See Janssen & Brumby (2010) for more detailed explanation.
## Lateral velocity update depends on current position in lane. Intuition behind function: the further away you are, the stronger the correction will be that a human makes
def vehicleUpdateActiveSteering(LD):

    latVel = 0.2617 * LD*LD + 0.0233 * LD - 0.022
    #Check if LD is positive/negative. positive (right of center) = steer left to correct
    if (LD > 0):
        latVel = -latVel
    returnValue = velocityCheckForVectors(latVel)
    return returnValue ### in m/s
	



### function to update lateral deviation in cases where the driver is NOT steering actively (when they are distracted by typing for example). Draw a value from a random distribution. This can be added to the position where the car is already.
def vehicleUpdateNotSteering():
    
    global gaussDeviateMean
    global gaussDeviateSD 

    

    vals = numpy.random.normal(loc=gaussDeviateMean, scale=gaussDeviateSD,size=1)[0]
    returnValue = velocityCheckForVectors(vals)
    return returnValue   ### in m/s





### Function to run a trial. Needs to be defined by students (section 2 and 3 of assignment)

def runTrial(nrWordsPerSentence =5,nrSentences=3,nrSteeringMovementsWhenSteering=2, interleaving="word", index=0):
    resetParameters()
    locPos = []             #stores all calculated position values
    trialTime = 0           #stores current trial time
    locColor = []           #stores colors to plot lane position r = no active steering; b = active steering
    locPos.append(startingPositionInLane) #start of trial position
    locColor.append("b") #otherwise you get an inconsistent amount of elements.
    typingSpeed = numpy.random.normal(loc=wordsPerMinuteMean, scale=wordsPerMinuteSD, size=1)[0]  # typing speed in words per minute
    global timePerWord
    timePerWord = (60 * 1000) / typingSpeed  # ms per word

    if(interleaving == "word"):
        for s in range(nrSentences):
            for w in range(nrWordsPerSentence):
                if(w == 0): #is it the first word?
                    timeTypedPerWord = timePerWord + retrievalTimeWord + retrievalTimeSentence
                else:
                    timeTypedPerWord = timePerWord + retrievalTimeWord
                numDriftSteps = int(timeTypedPerWord // timeStepPerDriftUpdate) #number of drifts during time typed rounded off
                #update locPos when not actively steering
                for step in range(numDriftSteps):   #updates the trial time per drift step
                    driftVel = vehicleUpdateNotSteering()
                    newPos = locPos[-1] + driftVel * (timeStepPerDriftUpdate/1000)
                    locPos.append(newPos)
                    trialTime += timeStepPerDriftUpdate
                    locColor.append("r")
                #Update locPos when actively steering (not last word)
                if not (s==nrSentences-1 and w == nrWordsPerSentence-1):
                    for steer in range(nrSteeringMovementsWhenSteering):
                        latVel = vehicleUpdateActiveSteering(locPos[-1]) #lateral velocity based on current lateral position
                        numSteerDriftSteps = int(steeringUpdateTime//timeStepPerDriftUpdate)    #makes sure to update per 50ms
                        for step in range(numSteerDriftSteps):
                            newPos = locPos[-1] + latVel * (timeStepPerDriftUpdate/1000)
                            locPos.append(newPos)
                            trialTime += timeStepPerDriftUpdate
                            locColor.append("b")
    elif(interleaving == "sentence"):
        for s in range(nrSentences):
            timeTypedPerSentence = retrievalTimeSentence
            for w in range(nrWordsPerSentence):
                timeTypedPerSentence += timePerWord
            numDriftSteps = int(timeTypedPerSentence // timeStepPerDriftUpdate)  # number of drifts during time typed rounded off
            # update locPos when not actively steering
            for step in range(numDriftSteps):  # updates the trial time per drift step
                driftVel = vehicleUpdateNotSteering()
                newPos = locPos[-1] + driftVel * (timeStepPerDriftUpdate / 1000)
                locPos.append(newPos)
                trialTime += timeStepPerDriftUpdate
                locColor.append("r")
            # Update locPos when actively steering (not last sentence)
            if not (s == nrSentences - 1):
                for steer in range(nrSteeringMovementsWhenSteering):
                    latVel = vehicleUpdateActiveSteering(locPos[-1])  # lateral velocity based on current lateral position
                    numSteerDriftSteps = int(steeringUpdateTime // timeStepPerDriftUpdate)  # makes sure to update per 50ms
                    for step in range(numSteerDriftSteps):
                        newPos = locPos[-1] + latVel * (timeStepPerDriftUpdate / 1000)
                        locPos.append(newPos)
                        trialTime += timeStepPerDriftUpdate
                        locColor.append("b")

    elif(interleaving == "drivingOnly"):
        for s in range(nrSentences):
            timeTypedPerSentence = retrievalTimeSentence
            for w in range(nrWordsPerSentence):
                timeTypedPerSentence += timePerWord
            numDriftSteps = int(timeTypedPerSentence // timeStepPerDriftUpdate)  # number of drifts during time typed rounded off
            # Update locPos when actively steering (not last word)
            if not (s == nrSentences - 1):
                for steer in range(nrSteeringMovementsWhenSteering):
                    latVel = vehicleUpdateActiveSteering(locPos[-1])  # lateral velocity based on current lateral position
                    numSteerDriftSteps = int(steeringUpdateTime // timeStepPerDriftUpdate)  # makes sure to update per 50ms
                    numDriftStepsUpdate =  int(numDriftSteps / numSteerDriftSteps)
                    for step in range(numDriftStepsUpdate):
                        newPos = locPos[-1] + latVel * (timeStepPerDriftUpdate / 1000)
                        locPos.append(newPos)
                        trialTime += timeStepPerDriftUpdate
                        locColor.append("b")


    else:  #interleaving = none
        for s in range(nrSentences):
            timeTypedPerSentence = retrievalTimeSentence
            for w in range(nrWordsPerSentence):
                timeTypedPerSentence += timePerWord
            numDriftSteps = int(timeTypedPerSentence // timeStepPerDriftUpdate)  # number of drifts during time typed rounded off
            # update locPos when not actively steering
            for step in range(numDriftSteps):  # updates the trial time per drift step
                driftVel = vehicleUpdateNotSteering()
                newPos = locPos[-1] + driftVel * (timeStepPerDriftUpdate / 1000)
                locPos.append(newPos)
                trialTime += timeStepPerDriftUpdate
                locColor.append("r")


    meanDeviation = numpy.mean(numpy.abs(locPos))   #Mean lane deviation
    maxDeviation = numpy.max(numpy.abs(locPos))     #max. lane deviation
    timeVector = numpy.arange(0, len(locPos) * timeStepPerDriftUpdate, timeStepPerDriftUpdate) #step per 50ms
    plt.figure(figsize=(10,7))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.scatter(timeVector, locPos, c=locColor)
    plt.xlabel("time (ms)")
    plt.ylabel("lane position (m)")
    plt.title(f"Lane position vs time (interleaving: {interleaving}, trial: {index})")
    summary_text = f"Total trial time: {trialTime:.2f} ms\nMean position on the road: {meanDeviation:.3f} m\nMax position on the road (Absolute): {maxDeviation:.3f} m"
    plt.text(0.05, 0.95, summary_text,
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.scatter([], [], c="b", label="Active steering")
    plt.scatter([], [], c="r", label="Typing")
    plt.legend()
    plt.show()
    return trialTime, locPos, locColor, meanDeviation, maxDeviation

# runTrial(nrWordsPerSentence=17, nrSentences=10,nrSteeringMovementsWhenSteering=4,interleaving='sentence')
runTrial(nrWordsPerSentence=17, nrSentences=10,nrSteeringMovementsWhenSteering=4,interleaving='drivingOnly')

# for i in range(10):
#     runTrial(nrWordsPerSentence=17, nrSentences=10,nrSteeringMovementsWhenSteering=4,interleaving='sentence',index=i+1)

### function to run multiple simulations. Needs to be defined by students (section 3 of assignment)
def runSimulations(nrSims = 100):
    print("hello world")



	




