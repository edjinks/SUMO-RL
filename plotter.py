import matplotlib.pyplot as plt
import numpy as np

############RESULTS PLOTTING######################

def plotArr(array, title):
    x = np.array([x for x in range(0,len(array))])
    y = np.array(array)
    plt.plot(x, y, 'o')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x+b)
    plt.title(title)
    plt.show()

def plotHistogram(arr, title):
    bins = 100
    plt.hist(arr, bins)
    plt.title(title)
    plt.show()


def compareHistograms(arrays, titles):
    bins = np.histogram(np.hstack((arrays[0],arrays[1])), bins=50)[1]
    for i in range(len(arrays)):
        plt.hist(arrays[i], bins, alpha=0.5, label=titles[i])
    plt.legend(loc='upper right')
    title = " & ".join(titles)
    plt.title(title)
    plt.show()

