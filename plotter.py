import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

############RESULTS PLOTTING FUNCTIONS######################
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
    
def plot2XArr(arrays, labels, title, xlab, ylab):
    fig, ax = plt.subplots()
    colors = plt.get_cmap('tab20').colors
    for i in range(2,len(arrays)):
        y = np.array(arrays[i])
        x = np.array([x*10 for x in range(0,len(arrays[i]))])
        ax.plot(x, y, 'o', color=colors[i])
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x+b, label=labels[i], color=colors[i])
    ax2 = ax.twinx()
    y = np.array(arrays[0])
    y2=np.array(arrays[1])
    ax2.plot(x, y, 'o', color=colors[0])
    ax2.plot(x, y2, 'o', color=colors[1])
    m, b = np.polyfit(x, y, 1)
    m2, b2 = np.polyfit(x, y2, 1)
    ax2.plot(x, m*x+b, label=labels[0], color=colors[0])
    ax2.plot(x, m2*x+b2, label=labels[1], color=colors[1])
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax2.set_ylabel('Avg Wait Times (s)')
    ax.set_title(title)
    fig.legend(loc='lower middle', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
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

def violin(arr):
    plt.violinplot(arr, showmeans=True, showextrema=True)
    plt.show()

def snsViolin(arr, name):
    dfdict =[]
    for i in range(len(arr)):
        for time in arr[i]:
            dfdict.append({'policy':name[i], 'wait':time})

    df = pd.DataFrame(dfdict)
    print(df)
    f, ax = plt.subplots(figsize=(8, 8))

    # Show each distribution with both violins and points
    sns.violinplot(x="policy",y="wait",data=df, inner="box", palette="Set3", cut=2, linewidth=3)

    sns.despine(left=True)

    f.suptitle('Waiting Time by Policy', fontsize=18, fontweight='bold')
    ax.set_xlabel("Policy",size = 16,alpha=0.7)
    ax.set_ylabel("Wait",size = 16,alpha=0.7)
    plt.show()

def snsBar(arr, name):
    dfdict =[]
    for i in range(len(arr)):
        for time in arr[i]:
            dfdict.append({'policy':name[i], 'wait':time})

    df = pd.DataFrame(dfdict)
    print(df)
    f, ax = plt.subplots(figsize=(8, 8))

    # Show each distribution with both violins and points
    sns.barplot(x="policy",y="wait")

    sns.despine(left=True)

    f.suptitle('Waiting Time by Policy', fontsize=18, fontweight='bold')
    ax.set_xlabel("Policy",size = 16,alpha=0.7)
    ax.set_ylabel("Wait",size = 16,alpha=0.7)
    plt.show()