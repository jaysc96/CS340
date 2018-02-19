import numpy as np
import pylab as plt
import kmeans
import utils

def quantizeImage (I, b):
    N, M, C = I.shape
    B = 2**b
    i = np.zeros(I.shape)
    for n in range(N):
        model = kmeans.fit(I[n,:,:], B)
        y = model['predict'](model, I[n,:,:])
        means = model['means']
        for bb in range(B):
            i[n, y==bb] = means[bb]
    return i