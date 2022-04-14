import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

############RESULTS PLOTTING######################
def pltRewardAndSteps(rewards, steps):
    x = np.array([x for x in range(0,len(rewards))])
    y = np.array(rewards)
    y2 = np.array(steps)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward", color="red")
    ax.set_title('Avg Reward')
    ax2 = ax.twinx()
    ax2.set_ylabel("Average Steps", color='blue')
    m, b = np.polyfit(x, y, 1)
    m2, b2 = np.polyfit(x, y2, 1)
    ax.plot(x, y, 'ro')
    ax2.plot(x, y2, 'bo')
    ax.plot(x, m*x+b, color='red')
    ax2.plot(x, m2*x+b2, color='blue')
    plt.show()

def plotArr(array, title):
    x = np.array([x for x in range(0,len(array))])
    y = np.array(array)

    plt.plot(x, y, 'o')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x+b)
    plt.title(title)
    plt.show()

def plotHistogram(arr, title):
    bins = 50
    plt.hist(arr, bins)
    plt.title(title)
    plt.show()

def compareHistograms(arr1, arr2, title1, title2):
    #bins = 100
    bins = np.histogram(np.hstack((arr1,arr2)), bins=100)[1]
    plt.hist(arr1, bins, alpha=0.5, label=title1)
    plt.hist(arr2, bins, alpha=0.5, label=title2)
    plt.legend(loc='upper right')
    plt.title(title1+' & '+title2)
    plt.show()


