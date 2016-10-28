__author__ = 'kezi'

import matplotlib.pyplot as plt
import numpy as np

count = np.load('count.npy')
count_true = np.load('count_true.npy')
for d in range(count.shape[0]):
    for c in range(count.shape[1]):
        plt.figure(figsize=(16, 6))
        markerline, stemlines, baseline = plt.stem(count[d, c, :])
        plt.setp(stemlines, 'linestyle', 'None')
        plt.plot(count_true[d, c, :], 'r')
        # plt.title('number of points in cluster %i of time series %i' % (c+1, d+1))
        plt.xlabel('time')
        plt.ylabel('number of samples in cluster %i' % (c+1))
        plt.savefig('num_points_d%ic%i' % (d+1, c+1))
