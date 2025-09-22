import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than "
                         "window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming',
                      'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', "
                         "'bartlett', 'blackman'")
    s = np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    # moving average
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]

protocols = ['vdw', 'bonded', 'coul']
for p in protocols:
    for l in list(range(0,12)):
        lam = l
        xs,ys,dhdls,xs2 = [],[],[],[]
        for r in [1,2,3]:
            file = os.path.join('%i/%s'%(r,p), 'dhdl_%s.pickle'%l)
            a = open(file, 'rb')
            dhdl = pickle.load(a)
            a.close()
            x = [i/1000 for i in list(range(0,len(dhdl)))]
            plt.plot(x, smooth(np.array(dhdl),window_len=100),'-o', alpha=0.4,markersize=1, label='repeat %i'%r)
        plt.title('%s lambda %i'%(p,l))
        plt.ylabel('dH/dl')
        plt.xlabel('simulation time in ns')
        plt.legend()
       	#plt.savefig('dHdl_%s.pdf', bbox_inches='tight')
        plt.show()
