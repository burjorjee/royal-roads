import numpy as np
import numpy.random as npr
import mmh3
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from joblib import Parallel, delayed

rng = npr.RandomState()

def generateContingentParitiesFunction(order, height):
    def contingentParitiesFunction(pop, verbose=False):
        assert(pop.shape[1] == order * height)
        popMissteps = []
        traceAndFitness = []
        for c in xrange(pop.shape[0]):
            output = 0
            ctr = 0
            length = pop.shape[1]
            loci = np.arange(length)
            missteps = []
            trace = ""
            while ctr < height:
                rng.seed(abs(mmh3.hash(trace)))
                acc = 0
                trace += "|"
                for i in xrange(order):
                    idx = rng.randint(length - (ctr * order + i)) + 1
                    swap = loci[-idx]
                    loci[-idx] = loci[ctr * order + i]
                    loci[ctr * order + i] = swap
                    trace += "%2d:%s|" % (swap + 1, int(pop[c, swap]))
                    acc += pop[c, swap]
                output += acc % 2

                if acc % 2 == 0:
                    missteps.append(ctr + 1)

                ctr +=1
            popMissteps.append(missteps)
            traceAndFitness.append((trace, height - len(missteps)))
        if verbose:
            for t in sorted(traceAndFitness):
                print "%s   %s " % t
        return np.array([height - len(missteps) for missteps in popMissteps]), popMissteps

    return contingentParitiesFunction


def evolve(fitnessFunction,
           length,
           popSize,
           maxGens,
           probMutation,
           sigmaScaling=True,
           sigmaScalingCoeff=1.0,
           useClamping=True,
           rngSeed=1,
           visualize=True):
    """

    :param fitnessFunction: the fitness function
    :param length: length of a chromosome
    :param popSize: the population size
    :param maxGens: the total number of generations in a run
    :param probMutation: the per locus probability of mutation
    :param sigmaScaling: is fitness sigma scaled? (True or False)
    :param sigmaScalingCoeff: the sigma scaling coefficient (lower => greater selection pressure
    :param useClamping: is clamping (http://s3.amazonaws.com/burjorjee/www/hyperclimbing_hypothesis_2013.pdf) used (True or False)
    :param rngSeed: the random number generator seed
    :param visualize: visualize the run (True or False)
    :return: conditionally return the maximal fitness achieved in each generation
    """

    npr.seed(rngSeed)
    flagFreq=0.01
    unflagFreq=0.1
    flagPeriod=60
    flaggedGens = -np.ones(length)

    avgFitnessHist = np.zeros(maxGens+1)
    maxFitnessHist = np.zeros(maxGens+1)
    minMisstepsHist = []

    pop = np.zeros((popSize, length), 'bool')
    pop[npr.rand(popSize, length) < 0.5] = 1
    for gen in xrange(maxGens+1):

        fitnessVals, missteps = fitnessFunction(pop)
        fitnessVals = np.transpose(fitnessVals)
        maxIndex = fitnessVals.argmax()
        maxFitnessHist[gen] = fitnessVals[maxIndex]
        minMisstepsHist.append(missteps[maxIndex])
        avgFitnessHist[gen] = fitnessVals.mean()
        sigma = np.std(fitnessVals)

        if visualize:
            visualizeRun(avgFitnessHist, maxFitnessHist, gen=gen)
            visualizeMissteps("generations", maxFitnessHist, minMisstepsHist, gen=gen)
            plt.ylabel('generations')

        print "\ngen = %.3d   avg fitness = %3.3f   fitness std = %3.3f   maxfitness = %3.3f" % (gen, avgFitnessHist[gen], sigma, maxFitnessHist[gen])

        if sigmaScaling:
            if sigma:
                fitnessVals = np.maximum(1 + (fitnessVals - fitnessVals.mean()) / (sigmaScalingCoeff * sigma), 0)
                fitnessVals[fitnessVals < 0] = 0
            else:
                fitnessVals = np.ones(popSize)

        # implement fitness proportional selection
        cumNormFitnessVals = np.cumsum(fitnessVals) / fitnessVals.sum()
        markers = npr.rand(2 * popSize)
        markers = np.sort(markers)
        parentIndices = np.zeros(2 * popSize, dtype='int16')
        ctr = 0
        for idx in xrange(2 * popSize):
            while markers[idx] > cumNormFitnessVals[ctr]:
                ctr += 1
            parentIndices[idx] = ctr
        npr.shuffle(parentIndices)

        # deterimine the first parents of each mating pair
        firstParents = pop[parentIndices[:popSize], :]
        # determine the second parents of each mating pair
        secondParents = pop[parentIndices[popSize:], :]

        crossoverMasks = npr.rand(popSize, length) < 0.5

        pop = firstParents
        pop[crossoverMasks] = secondParents[crossoverMasks]

        bitFreqs = pop.sum(axis=0).astype('float')/popSize
        if visualize and gen % 10 == 0:
            visualizeGen(bitFreqs, gen=gen, avgFitness=avgFitnessHist[gen], maxFitness=maxFitnessHist[gen])

        # Do not mutate loci that have been clamped
        mutationMasks = npr.rand(popSize, length) < probMutation
        if useClamping:
            flaggedGens[0.5 - abs(0.5 - bitFreqs) > unflagFreq] = -1
            flaggedGens[np.logical_and(0.5 - abs(0.5 - bitFreqs) < flagFreq, flaggedGens < 0)] = 0
            flaggedGens[flaggedGens >= 0] += 1
            mutateLocus = flaggedGens <= flagPeriod
            x = flaggedGens[mutateLocus]
            print ' FlaggedLoci = %s, minDistToThresplt.hold = %s, unMutatedLoci = %s' % \
                  (sum(flaggedGens > 0), flagPeriod - max(x) + 1 if x != np.array([]) else "NA", sum(np.logical_not(mutateLocus)))
            mutationMasks[:, np.logical_not(mutateLocus)] = False

        pop[mutationMasks] = np.logical_not(pop[mutationMasks])

    fitnessVals, missteps = fitnessFunction(pop)
    fitnessVals = np.transpose(fitnessVals)
    maxIndex = fitnessVals.argmax()
    maxFitnessHist[gen] = fitnessVals[maxIndex]
    minMisstepsHist.append(missteps[maxIndex])
    avgFitnessHist[gen] = fitnessVals.mean()
    if visualize:
        visualizeRun(avgFitnessHist, maxFitnessHist, gen=gen, force=True)
        visualizeMissteps("generations", maxFitnessHist, minMisstepsHist, gen=gen, force=True)
    else:
        return maxFitnessHist, minMisstepsHist


def anneal(fitnessFunction,
           length,
           epochsPerPeriod,
           numPeriods,
           initFitnessDropOfOneAcceptanceProb,
           finalFitnessDropOfOneAcceptanceProb,
           rngSeed=1,
           visualize=True):
    """

    :param fitnessFunction: the fitness function
    :param length: length of a chromosome
    :param epochsPerPeriod: epochs per period
    :param numPeriods: total number of periods in a run
    :param initFitnessDropOfOneAcceptanceProb: initial probability of accepting a new candidate solution with a fitness delta of -1
    :param finalFitnessDropOfOneAcceptanceProb: final probability of accepting a new candidate solution with a fitness delta of -1
    :param rngSeed: the random number generator seed
    :param visualize: visualize the run (True or False)
    :return: conditionally return the maximal fitness achieved in each period
    """

    npr.seed(rngSeed)
    x = npr.rand(1, length) < 0.5

    fitnessVals, missteps = fitnessFunction(x)
    v = fitnessVals[0]
    ms = missteps[0]
    Tmax = - 1 / math.log(initFitnessDropOfOneAcceptanceProb)
    Tmin = - 1 / math.log(finalFitnessDropOfOneAcceptanceProb)
    Tfactor = -math.log(Tmax / Tmin)
    maxFitnessHist = np.zeros(numPeriods + 1)
    minMisstepsHist = [None] * (numPeriods + 1)
    for p in xrange(numPeriods+1):
        for i in xrange(epochsPerPeriod):

            m = int(npr.random()*length)
            x[0, m] ^= True
            fitnessVals, missteps = fitnessFunction(x)
            v_new = fitnessVals[0]
            ms_new = missteps[0]
            T = Tmax * math.exp(Tfactor * (p * epochsPerPeriod + i) / (epochsPerPeriod * numPeriods))
            dE = v_new - v
            fitnessIncreased = False
            if dE <= 0.0:
                if npr.random() < math.exp(dE / T):
                    v = v_new
                    ms = ms_new
                else:
                    # rollback
                    x[0, m] ^= True
            else:
                v = v_new
                ms = ms_new
                fitnessIncreased = True

            if fitnessIncreased or i == 0:
                maxFitnessHist[p] = v
                minMisstepsHist[p] = ms

        print u"\nperiod %5d, T = %1.5f, p(\u0394=-1) = %1.5f, value = %s " % (p, T, math.exp(-1 / T), maxFitnessHist[p])
        if visualize:
            visualizeMissteps("periods", maxFitnessHist, minMisstepsHist, gen=p)
    if visualize:
        visualizeMissteps("periods", maxFitnessHist, minMisstepsHist, gen=p, force=True)
    else:
        return maxFitnessHist, minMisstepsHist


def visualizeGen(bitFreqs, gen, avgFitness, maxFitness):
    length = len(bitFreqs)
    f = plt.figure(1)
    plt.hold(False)
    plt.plot(np.arange(length), bitFreqs,'b.', markersize=10, color='#3030ff')
    plt.axis([0, length, 0, 1])
    plt.title("Generation = %s, Average Fitness = %0.3f, Max Fitness = %0.3f" % (gen, avgFitness, maxFitness))
    plt.ylabel('Frequency of the Bit 1')
    plt.xlabel('Locus')
    f.canvas.draw()
    f.show()

def visualizeRun(avgFitnessHist, maxFitnessHist, gen=None, force=False):
    if gen % 10 == 1 or force:
        f = plt.figure(2)
        plt.hold(False)
        plt.plot(np.arange(gen), avgFitnessHist[:gen] if gen else avgFitnessHist, 'k-')
        plt.hold(True)
        plt.plot(np.arange(gen), maxFitnessHist[:gen] if gen else maxFitnessHist, 'c-')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        f.canvas.draw()
        f.show()

def visualizeMissteps(xLabel, maxFitnessHist, minMisstepsHist, gen, force=False):
    f = plt.figure(3)
    if gen == 0:
        plt.clf()
        plt.hold(False)
    plt.plot(np.ones(len(minMisstepsHist[gen])) * gen, minMisstepsHist[gen], '.', color='#8080ff')
    if gen == 0:
        ax = f.axes[0]
        bx = ax.twinx()

    plt.hold(True)
    if gen % 10 == 0 or force:
        plt.plot(np.arange(gen), maxFitnessHist[:gen], 'r-')
        ax, bx = f.axes
        ax.set_ylabel('missteps', color='b')
        bx.set_ylabel('fitness', color='r', rotation=270)
        bx.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel(xLabel)
        f.canvas.draw()
        f.show()

def GA(rngSeed, numGenerations, useClamping=False, visualize=True):
    return evolve(fitnessFunction=generateContingentParitiesFunction(height=100, order=2),
                  length=200,
                  popSize=500,
                  maxGens=numGenerations,
                  probMutation=0.004,
                  sigmaScaling=True,
                  sigmaScalingCoeff=0.5,
                  useClamping=useClamping,
                  rngSeed=rngSeed,
                  visualize=visualize)

def SA(rngSeed, numPeriods, visualize=True):
    return anneal(fitnessFunction=generateContingentParitiesFunction(height=100, order=2),
                  length=200,
                  epochsPerPeriod=500,
                  numPeriods=numPeriods,
                  initFitnessDropOfOneAcceptanceProb=0.6,
                  finalFitnessDropOfOneAcceptanceProb=0.001,
                  rngSeed=rngSeed,
                  visualize=visualize)

def compareAlgorithms(numRuns):
    f = plt.figure(5)
    plt.clf()
    plt.hold(True)

    maxFitnessHists, minMisstepsHists = zip(*Parallel(n_jobs=-1)(delayed(GA)(i, 1000, True, False) for i in range(numRuns)))
    maxFitnessHists = np.array(maxFitnessHists)

    stdDev = maxFitnessHists.std(axis=0)
    avg = maxFitnessHists.mean(axis=0)
    plt.plot(np.arange(len(avg)), avg, color='g')
    plt.fill_between(np.arange(len(avg)), avg - stdDev, avg + stdDev, facecolor='g', alpha=0.2)
    m = maxFitnessHists

    maxFitnessHists, minMisstepsHists = zip(*Parallel(n_jobs=-1)(delayed(SA)(i, 1000, False) for i in range(numRuns)))
    maxFitnessHists = np.array(maxFitnessHists)

    stdDev = maxFitnessHists.std(axis=0)
    avg = maxFitnessHists.mean(axis=0)
    plt.plot(np.arange(len(avg)), avg, color='m', label= "Simulated annealing")
    plt.fill_between(np.arange(len(avg)), avg - stdDev, avg + stdDev, facecolor='m', alpha=0.2)

    plt.xlabel('generations / periods')
    plt.ylabel('fitness')
    green_patch = patches.Patch(color='green', label='Genetic Algorithm')
    purple_patch = patches.Patch(color='magenta', label='Simulated Annealing')
    plt.legend(handles=[green_patch, purple_patch], loc='upper left')
    f.canvas.draw()
    f.show()
    return m, maxFitnessHists
