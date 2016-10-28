import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle


segLens = [40, 80, 120]
qs = [2]

pH = np.load('pH.npy')
idx = np.load('./index/idx.npy')
for segLen in segLens:
    for q in qs:
        # load data
        feats = pickle.load(open('./data/feats_time_freq_whole_seg.pickle', 'rb'))
        data = feats[segLen]  # still a dictionary!
        # transform dic to array
        duration = 45 * 60 * 4  # use 45-min data
        numSeq = len(idx) / 2
        numSeg = duration / segLen
        data_healthy = np.zeros([numSeq, numSeg, 14])
        data_unhealthy = np.zeros([numSeq, numSeg, 14])
        for i, ind in enumerate(idx[0:numSeq]):
            data_unhealthy[i, :, :] = data[ind][-numSeg:]
        for i, ind in enumerate(idx[numSeq:]):
            data_healthy[i, :, :] = data[ind][-numSeg:]

        pca = PCA(n_components=q)
        scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        init = 30 * 60 * 4 / segLen
        data_after_pca = np.empty((numSeg-init, numSeq*2, init, q))
        for x in xrange(init, numSeg):
            data_fit = np.concatenate((data_unhealthy[:, x-init:x, :], data_healthy[:, x-init:x, :]), axis=0)
            shape = data_fit.shape
            data_reshape = np.reshape(data_fit, (shape[0] * shape[1], shape[2]))
            data_reshape_scaled = scalar.fit_transform(data_reshape)
            data_reshape_scaled_pca = pca.fit_transform(data_reshape_scaled)
            data_pca = np.reshape(data_reshape_scaled_pca, (shape[0], shape[1], q))
            data_after_pca[x-init, :, :, :] = data_pca
        sio.savemat('./matlab/data_after_pca_%ds_dim%d.mat' % (segLen/4, q), mdict={'data':data_after_pca})
