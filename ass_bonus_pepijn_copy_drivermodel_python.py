import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# --- Global Parameters --- #
steeringUpdateTime = 250
timeStepPerDriftUpdate = 50
startingPositionInLane = 0.27

gaussDeviateMean = 0
gaussDeviateSD = 0.13

maxLateralVelocity = 1.7
minLateralVelocity = -maxLateralVelocity
startvelocity = 0

retrievalTimeWord = 200
retrievalTimeSentence = 300

timePerWord = 0
wordsPerMinuteMean = 39.33
wordsPerMinuteSD = 10.3


def resetParameters():
    global timePerWord, retrievalTimeWord, retrievalTimeSentence
    global steeringUpdateTime, startingPositionInLane
    global gaussDeviateMean, gaussDeviateSD
    global maxLateralVelocity, minLateralVelocity, startvelocity
    global wordsPerMinuteMean, wordsPerMinuteSD
    timePerWord = 0
    retrievalTimeWord = 200
    retrievalTimeSentence = 300
    steeringUpdateTime = 250
    startingPositionInLane = 0.27
    gaussDeviateMean = 0
    gaussDeviateSD = 0.13
    maxLateralVelocity = 1.7
    minLateralVelocity = -1.7
    startvelocity = 0
    wordsPerMinuteMean = 39.33
    wordsPerMinuteSD = 10.3


def velocityCheckForVectors(v):
    if isinstance(v, list):
        return [max(min(x, maxLateralVelocity), minLateralVelocity) for x in v]
    else:
        return max(min(v, maxLateralVelocity), minLateralVelocity)


def vehicleUpdateActiveSteering(LD):
    latVel = 0.2617 * LD*LD + 0.0233 * LD - 0.022
    if LD > 0:
        latVel = -latVel
    return velocityCheckForVectors(latVel)


def vehicleUpdateNotSteering():
    val = np.random.normal(gaussDeviateMean, gaussDeviateSD)
    return velocityCheckForVectors(val)


# --- Run a single trial (faster version, no plotting) --- #
def runTrial(nrWordsPerSentence=5, nrSentences=3, nrSteeringMovementsWhenSteering=2, interleaving="word"):
    resetParameters()
    locPos = [startingPositionInLane]
    trialTime = 0

    # typing speed in ms per word
    typingSpeed = np.random.normal(wordsPerMinuteMean, wordsPerMinuteSD)
    global timePerWord
    timePerWord = 60000 / typingSpeed

    # --- Interleaving logic --- #
    if interleaving == "word":
        for s in range(nrSentences):
            for w in range(nrWordsPerSentence):
                timeTyped = timePerWord + retrievalTimeWord
                if w == 0:
                    timeTyped += retrievalTimeSentence
                numDriftSteps = int(timeTyped // timeStepPerDriftUpdate)
                # car drifts while typing
                for _ in range(numDriftSteps):
                    driftVel = vehicleUpdateNotSteering()
                    newPos = locPos[-1] + driftVel * (timeStepPerDriftUpdate/1000)
                    locPos.append(newPos)
                    trialTime += timeStepPerDriftUpdate
                # active steering after each word except last
                if not (s == nrSentences-1 and w == nrWordsPerSentence-1):
                    for _ in range(nrSteeringMovementsWhenSteering):
                        latVel = vehicleUpdateActiveSteering(locPos[-1])
                        numSteerSteps = int(steeringUpdateTime // timeStepPerDriftUpdate)
                        for _ in range(numSteerSteps):
                            newPos = locPos[-1] + latVel*(timeStepPerDriftUpdate/1000)
                            locPos.append(newPos)
                            trialTime += timeStepPerDriftUpdate

    elif interleaving == "sentence":
        for s in range(nrSentences):
            timeTyped = retrievalTimeSentence + nrWordsPerSentence*timePerWord
            numDriftSteps = int(timeTyped // timeStepPerDriftUpdate)
            for _ in range(numDriftSteps):
                driftVel = vehicleUpdateNotSteering()
                newPos = locPos[-1] + driftVel*(timeStepPerDriftUpdate/1000)
                locPos.append(newPos)
                trialTime += timeStepPerDriftUpdate
            if not s == nrSentences-1:
                for _ in range(nrSteeringMovementsWhenSteering):
                    latVel = vehicleUpdateActiveSteering(locPos[-1])
                    numSteerSteps = int(steeringUpdateTime // timeStepPerDriftUpdate)
                    for _ in range(numSteerSteps):
                        newPos = locPos[-1] + latVel*(timeStepPerDriftUpdate/1000)
                        locPos.append(newPos)
                        trialTime += timeStepPerDriftUpdate

    elif interleaving == "drivingOnly":
        totalTrialTime = 0
        for s in range(nrSentences):
            totalTrialTime += retrievalTimeSentence + nrWordsPerSentence*timePerWord
        numDriftSteps = int(totalTrialTime // timeStepPerDriftUpdate)
        for _ in range(numDriftSteps):
            latVel = vehicleUpdateActiveSteering(locPos[-1])
            newPos = locPos[-1] + latVel*(timeStepPerDriftUpdate/1000)
            locPos.append(newPos)
            trialTime += timeStepPerDriftUpdate

    elif interleaving == "2-3 words":
        """
        This method is an implemenation of bonus exercise 4,
        interleaving after 2-3 words (choosen randomly)
        """
        for s in range(nrSentences):
            word_index = 0
            # check whether at end of sentence
            while word_index < nrWordsPerSentence:
                # random choice of interleaving after 2 or 3 words
                rand_amount_words = np.random.choice([2, 3])

                # check if # words doesn't exceed remaing words sentence
                amount_words_interleave = min(rand_amount_words, nrWordsPerSentence - word_index)
                
                # calculate typing/drift times
                timeTyped = retrievalTimeWord * amount_words_interleave  + timePerWord * amount_words_interleave 
                numDriftSteps = int(timeTyped // timeStepPerDriftUpdate)
                for _ in range(numDriftSteps):
                    driftVel = vehicleUpdateNotSteering()
                    newPos = locPos[-1] + driftVel * (timeStepPerDriftUpdate / 1000)
                    locPos.append(newPos)
                    trialTime += timeStepPerDriftUpdate
                
                # calculate time for steering back 
                for _ in range(nrSteeringMovementsWhenSteering):
                    latVel = vehicleUpdateActiveSteering(locPos[-1])
                    numSteerDriftSteps = int(steeringUpdateTime // timeStepPerDriftUpdate)
                    for _ in range(numSteerDriftSteps):
                        newPos = locPos[-1] + latVel * (timeStepPerDriftUpdate / 1000)
                        locPos.append(newPos)
                        trialTime += timeStepPerDriftUpdate
                
                # set index at new position
                word_index += amount_words_interleave 

    else:  # interleaving == "none"
        totalTrialTime = 0
        for s in range(nrSentences):
            totalTrialTime += retrievalTimeSentence + nrWordsPerSentence*timePerWord
        numDriftSteps = int(totalTrialTime // timeStepPerDriftUpdate)
        for _ in range(numDriftSteps):
            driftVel = vehicleUpdateNotSteering()
            newPos = locPos[-1] + driftVel*(timeStepPerDriftUpdate/1000)
            locPos.append(newPos)
            trialTime += timeStepPerDriftUpdate

    meanDeviation = np.mean(np.abs(locPos))
    maxDeviation = np.max(np.abs(locPos))
    return trialTime, meanDeviation, maxDeviation



# Run the simulations
def runSimulations(nrSims=25):
    conditions = ["word", "sentence", "drivingOnly", "none", "2-3 words"]
    totalTime = []
    meanDeviation = []
    maxDeviation = []
    Condition = []

    for cond in conditions:
        for _ in range(nrSims):
            nrWordsPerSentence = np.random.randint(15, 21)  # 15–20 words
            nrSentences = 10
            nrSteeringMovementsWhenSteering = 4
            trialTime, meanDev, maxDev = runTrial(
                nrWordsPerSentence, nrSentences, nrSteeringMovementsWhenSteering, interleaving=cond
            )
            totalTime.append(trialTime)
            meanDeviation.append(meanDev)
            maxDeviation.append(maxDev)
            Condition.append(cond)

    return totalTime, meanDeviation, maxDeviation, Condition


# Plot the results
def plotSimulations(totalTime, meanDeviation, maxDeviation, Condition):
    conditions = ["word", "sentence", "drivingOnly", "none", "2-3 words"]
    symbols = {"word": "o", "sentence": "^", "drivingOnly": "s", "none": "x", "2-3 words": "*"}
    colors = {"word": "blue", "sentence": "red", "drivingOnly": "green", "none": "black", "2-3 words": "pink"}

    plt.figure(figsize=(10,7))

    for cond in conditions:
        idx = [i for i, c in enumerate(Condition) if c == cond]

        # Raw trial points
        plt.scatter(
            [totalTime[i] for i in idx], 
            [maxDeviation[i] for i in idx], 
            c='grey', marker=symbols[cond], alpha=0.5
        )

        # Mean + error bars
        meanTime = np.mean([totalTime[i] for i in idx])
        meanMax = np.mean([maxDeviation[i] for i in idx])
        stdTime = np.std([totalTime[i] for i in idx])
        stdMax = np.std([maxDeviation[i] for i in idx])
        plt.errorbar(
            meanTime, meanMax, xerr=stdTime, yerr=stdMax, fmt=symbols[cond],
            c=colors[cond], markersize=10, capsize=5
        )

    plt.xlabel("Total trial time (ms)")
    plt.ylabel("Max lateral deviation (m)")
    plt.title("Simulation results (25 trials per condition)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # --- Legend using proxy artists --- #
    legend_elements = []
    for cond in conditions:
        # Raw trials
        legend_elements.append(
            mlines.Line2D([], [], color='grey', marker=symbols[cond], linestyle='None', label=f'{cond} trials')
        )
        # Means
        legend_elements.append(
            mlines.Line2D([], [], color=colors[cond], marker=symbols[cond], linestyle='None', markersize=10, label=f'{cond} mean ± std')
        )

    plt.legend(handles=legend_elements)
    plt.show()



# --- Example usage --- #
totalTime, meanDeviation, maxDeviation, Condition = runSimulations(nrSims=100)
plotSimulations(totalTime, meanDeviation, maxDeviation, Condition)
