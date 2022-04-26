from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

############RESULTS PLOTTING######################
def plotArr(array, title, xlab, ylab):
    x = np.array([x*10 for x in range(0,len(array))])
    y = np.array(array)
    plt.plot(x, y, 'o')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x+b)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title)
    plt.show()

def plotXArr(arrays, labels, title, xlab, ylab):
    fig, ax = plt.subplots()
    colors = plt.get_cmap('tab20').colors
    for i in range(1,len(arrays)):
        y = np.array(arrays[i])
        x = np.array([x*10 for x in range(0,len(arrays[i]))])
        ax.plot(x, y, 'o', color=colors[i])
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x+b, label=labels[i], color=colors[i])

    ax2 = ax.twinx()
    y = np.array(arrays[0])
    ax2.plot(x, y, 'o', color=colors[0])
    m, b = np.polyfit(x, y, 1)
    ax2.plot(x, m*x+b, label=labels[0], color=colors[0])

    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax2.set_ylabel('Avg Wait Times (s)')
    ax.set_title(title)
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    plt.show()
    

def compareHistograms(bins, arrays, titles, quantiles, plotQ, title):
    bins = np.histogram(np.hstack(arrays), bins=bins)[1]
    for i in range(len(arrays)):
        _,_,c = plt.hist(arrays[i], bins, alpha=0.5, label=titles[i])
        if plotQ:
            for q in np.percentile(arrays[i], quantiles):
                plt.axvline(q, alpha=1, color = c[0].get_facecolor())
    plt.legend(loc='upper right')
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Agent Waiting Time (s)')
    plt.show()

def plotBar(vals, labels):
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels)
    plt.title('Collisions Bar Chart')
    plt.ylabel('Number of Collisions')
    plt.show()