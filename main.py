import numpy as np
import pickle
from HCRFC_class import GibbsSampler
from model_loglikelihood import dict2mix, all_loglike
from sklearn.decomposition import PCA
from sklearn import preprocessing
import os


# specify parameters
segLens = [80, 120]
qs = [4]

for segLen in segLens:
    for q in qs:
        # load data
        feats = pickle.load(open('./data/feats_time_freq_whole_seg.pickle', 'rb'))
        pH = np.load('pH.npy')
        idx = np.load('./index/idx.npy')
        data = feats[segLen]        # still a dictionary!
        # transform dic to array
        duration = 45*60*4          # use 45-min data
        numSeq = len(idx)/2
        numSeg = duration/segLen
        data_healthy = np.zeros([numSeq, numSeg, 14])
        data_unhealthy = np.zeros([numSeq, numSeg, 14])
        for i, ind in enumerate(idx[0:numSeq]):
            data_unhealthy[i, :, :] = data[ind][-numSeg:]
        for i, ind in enumerate(idx[numSeq:]):
            data_healthy[i, :, :] = data[ind][-numSeg:]

        # Use 30 FHR sequences for training and 5 for testing.
        data_healthy_train = data_healthy[0:30]
        data_unhealthy_train = data_unhealthy[0:30]

        # perform PCA
        # We set the capacity of the model to be 30-min, thus the first 30-min training data will be used to fit the PCA
        # and the rest of data are transformed using the fitted PCA.
        # TODO: DO we update the PCA instance when new data coming in? Now no.
        # First fit the PCA instance
        pca = PCA(n_components=q)
        scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        data_fit = np.concatenate((data_healthy_train[:, 0:30*60*4/segLen, :], data_unhealthy_train[:, 0:30*60*4/segLen, :]),
                                  axis=0)
        shape = data_fit.shape
        # shape_transform = data_healthy_test.shape
        data_reshape = np.reshape(data_fit, (shape[0]*shape[1], shape[2]))
        data_reshape_scaled = scalar.fit_transform(data_reshape)
        pca.fit(data_reshape_scaled)        # fit the PCA instance with the first 30-min training data
        # Then transform all the data
        shape = data_healthy.shape
        data_reshape = np.reshape(data_healthy, (shape[0]*shape[1], shape[2]))
        data_reshape_scaled = scalar.transform(data_reshape)
        data_healthy_pca_reshape = pca.transform(data_reshape_scaled)
        data_healthy_pca = np.reshape(data_healthy_pca_reshape, (shape[0], shape[1], q))

        shape = data_unhealthy.shape
        data_reshape = np.reshape(data_unhealthy, (shape[0]*shape[1], shape[2]))
        data_reshape_scaled = scalar.transform(data_reshape)
        data_unhealthy_pca_reshape = pca.transform(data_reshape_scaled)
        data_unhealthy_pca = np.reshape(data_unhealthy_pca_reshape, (shape[0], shape[1], q))

        data_healthy_train = data_healthy_pca[0:30]
        data_healthy_test = data_healthy_pca[30:]
        data_unhealthy_train = data_unhealthy_pca[0:30]
        data_unhealthy_test = data_unhealthy_pca[30:]
        data_test = np.concatenate((data_healthy_test, data_unhealthy_test), axis=0)

        for run in xrange(1):
            # train the model
            folder = './results/time_freq_dim%d_%ds_%d-th_run/' % (q, segLen/4, run+1)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            hcrfc_hl = GibbsSampler(snapshot_interval=10)
            hcrfc_un = GibbsSampler(snapshot_interval=10)

            init = 30*60*4/segLen

            numIter = 100
            hcrfc_hl.initialize(data_healthy_train[:, 0:init, :])
            hcrfc_hl.sample(numIter)
            hcrfc_un.initialize(data_unhealthy_train[:, 0:init, :])
            hcrfc_un.sample(numIter)
            hcrfc_hl.pickle(folder, 'hcrfc_hl_init_%d_iter' % numIter); hcrfc_un.pickle(folder, 'hcrfc_un_init_%d_iter' % numIter)

            # classify current test data
            numTestSeq = data_test.shape[0]
            confidence = np.zeros((numTestSeq, numSeg-init+1))

            weights_hl, dists_hl = dict2mix(hcrfc_hl.params)
            weights_un, dists_un = dict2mix(hcrfc_un.params)

            p0 = np.array([all_loglike(data_test[i, 0:init, :], weights_hl, dists_hl) for i in xrange(numTestSeq)])
            p1 = np.array([all_loglike(data_test[i, 0:init, :], weights_un, dists_un) for i in xrange(numTestSeq)])
            pred = p0 < p1
            confidence[:, 0] = 100 * np.exp(p0/init) / (np.exp(p0/init) + np.exp(p1/init))
            print 'Initialization completed.'

            for n in xrange(init, numSeg):
                print 'Move to the sequence starting with the %d-th segment.' % n

                hcrfc_un.new_customer(data_unhealthy_train[:, n, :])
                hcrfc_un.sample(10)
                hcrfc_hl.new_customer(data_healthy_train[:, n, :])
                hcrfc_hl.sample(10)
                numIter += 10
                hcrfc_hl.pickle(folder, 'hcrfc_hl_%d_newdata_%d_iter' % (n-init+1, numIter))
                hcrfc_un.pickle(folder, 'hcrfc_un_%d_newdata_%d_iter' % (n-init+1, numIter))

                weights_hl, dists_hl = dict2mix(hcrfc_hl.params)
                weights_un, dists_un = dict2mix(hcrfc_un.params)

                p0 = np.array([all_loglike(data_test[i, n-init+1:n+1, :], weights_hl, dists_hl) for i in xrange(numTestSeq)])
                p1 = np.array([all_loglike(data_test[i, n-init+1:n+1, :], weights_un, dists_un) for i in xrange(numTestSeq)])

                confidence[:, n-init+1] = 100 * np.exp(p0 / init) / (np.exp(p0 / init) + np.exp(p1 / init))

            np.save(folder+'confidence', confidence)
            print '%d-th run completed.' % (run+2)
        print '%d seconds with %d dimension finished' % (segLen/4, q)
