import numpy as np

def plot_ax(ax, signal, sampling_rate):
        
    color_line = (0, 0, 0.7)    
    color_major = (1, 0, 0)
    color_minor = (1, 0.7, 0.7)
    
    signal = signal - (signal.max()+signal.min())/2 # center signal
    ax.plot(np.arange(0,len(signal)),signal,linewidth=0.5,color=color_line)

    # set major grid
    xmajor = np.arange(0,len(signal),sampling_rate*0.2)    
    ymajor = np.arange(-1, 1, 0.5)
    ax.set_xticks(xmajor, minor=False)
    ax.set_xticks(xmajor, minor=False)
    ax.set_yticks(ymajor, minor=False)    
    ax.grid(which="major", color=color_major, linewidth=0.5)
    
    # set minor grid
    xminor = np.arange(0, len(signal), sampling_rate*0.04)
    yminor = np.arange(-1, 1, 0.1)
    ax.set_xticks(xminor, minor=True)
    ax.set_yticks(yminor, minor=True)
    ax.grid(which="minor", color=color_minor, linewidth=0.5)
        
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    
    ax.margins(0)
        
    ax.set_ylim(-1,1)