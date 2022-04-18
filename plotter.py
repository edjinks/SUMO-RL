from cProfile import label
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

def compareHistograms(bins, arrays, titles, quantiles):
    bins = np.histogram(np.hstack(arrays), bins=bins)[1]
    for i in range(len(arrays)):
        _,_,c = plt.hist(arrays[i], bins, alpha=0.5, label=titles[i])
        for q in np.percentile(arrays[i], quantiles):
            plt.axvline(q, alpha=1, color = c[0].get_facecolor())
    plt.legend(loc='upper right')
    title = 'Agent Waiting Times using Different Driving Policies for 5000 Vehicles'
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Agent Waiting Time')
    plt.show()

