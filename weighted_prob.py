import numpy as np
import pickle
from model_loglikelihood import dict2mix, loglike_array, all_loglike
from sklearn.decomposition import PCA
from sklearn import preprocessing


segLen = 40
q = 4
run = 1

# process data
feats = pickle.load(open('./data/feats_time_freq_whole_seg.pickle', 'rb'))
delBeats = pickle.load(open('../../data/CTU-UHB/delBeats.pickle', 'rb'))
pH = np.load('pH.npy')
idx = np.load('./index/idx.npy')
idx_unhealthy_test = np.array([17, 35, 117, 120, 129, 146, 179, 186, 324, 328, 405, 465, 469, 493, 548])
data = feats[segLen]        # still a dictionary!
# transform dic to array
duration = 45*60*4          # use 45-min data
numSeq = len(idx)/2
numSeg = duration/segLen
data_healthy = np.zeros([numSeq, numSeg, 14])
data_unhealthy = np.zeros([numSeq, numSeg, 14])
delbeats = np.zeros([2*numSeq, duration])

data_test_new = np.zeros([len(idx_unhealthy_test), numSeg, 14])
delbeats_new = np.zeros([len(idx_unhealthy_test), duration])
for i, ind in enumerate(idx[0:numSeq]):
    data_unhealthy[i, :, :] = data[ind][-numSeg:]
    delbeats[i, :] = delBeats[ind][-duration:]
for i, ind in enumerate(idx[numSeq:]):
    data_healthy[i, :, :] = data[ind][-numSeg:]
    delbeats[i+numSeq, :] = delBeats[ind][-duration:]
for i, ind in enumerate(idx_unhealthy_test):
    data_test_new[i, :, :] = data[ind][-numSeg:]
    delbeats_new[i, :] = delBeats[ind][-duration:]

# Use 30 FHR sequences for training and 5 for testing.
data_healthy_train = data_healthy[0:30]
data_unhealthy_train = data_unhealthy[0:30]

# perform PCA
# We set the capacity of the model to be 30-min, thus the first 30-min training data will be used to fit the PCA,
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

shape = data_test_new.shape
data_reshape = np.reshape(data_test_new, (shape[0]*shape[1], shape[2]))
data_reshape_scaled = scalar.transform(data_reshape)
data_test_new_pca_reshape = pca.transform(data_reshape_scaled)
data_test_new_pca = np.reshape(data_test_new_pca_reshape, (shape[0], shape[1], q))

data_healthy_train = data_healthy_pca[0:30]
data_healthy_test = data_healthy_pca[30:]
data_unhealthy_train = data_unhealthy_pca[0:30]
data_unhealthy_test = data_unhealthy_pca[30:]

data_test = np.concatenate((data_healthy_test, data_unhealthy_test), axis=0)
delbeats_test = np.concatenate((delbeats[65:, :], delbeats[30:35, :]))

# data_test = data_test_new_pca
# delbeats_test = delbeats_new

numIter = 100
init = 30*60*4/segLen

folder = './results/time_freq_dim%d_%ds_%d-th_run/' % (q, segLen/4, run)
hcrfc_hl = pickle.load(open(folder + 'hcrfc_hl_init_%d_iter' % numIter))
hcrfc_un = pickle.load(open(folder + 'hcrfc_un_init_%d_iter' % numIter))

numTestSeq = data_test.shape[0]
unreweighted_confidence = np.zeros((numTestSeq, numSeg-init+1))
reweighted_confidence = np.zeros((numTestSeq, numSeg-init+1))

prob_weights = np.zeros([data_test.shape[0], numSeg])
for i in xrange(delbeats_test.shape[0]):
    prob_weights[i] = np.array([1 - 1.*np.sum(delbeats_test[i][segLen*j:segLen*(j+1)])/segLen for j in xrange(numSeg)])

weights_hl, dists_hl = dict2mix(hcrfc_hl.params)
weights_un, dists_un = dict2mix(hcrfc_un.params)

p0 = np.array([loglike_array(data_test[i, 0:init, :], weights_hl, dists_hl) for i in xrange(numTestSeq)])
p1 = np.array([loglike_array(data_test[i, 0:init, :], weights_un, dists_un) for i in xrange(numTestSeq)])

p0_unreweighted = np.array([all_loglike(data_test[i, 0:init, :], weights_hl, dists_hl) for i in xrange(numTestSeq)])
p1_unreweighted = np.array([all_loglike(data_test[i, 0:init, :], weights_un, dists_un) for i in xrange(numTestSeq)])

reweights = np.array([prob_weights[i, 0:init] / np.sum(prob_weights[i, 0:init]) for i in xrange(prob_weights.shape[0])])
p0_reweighted = np.sum([p0[i]*reweights[i] for i in xrange(numTestSeq)], axis=1)
p1_reweighted = np.sum([p1[i]*reweights[i] for i in xrange(numTestSeq)], axis=1)

unreweighted_confidence[:, 0] = 100 * np.exp(p0_unreweighted/init) / (np.exp(p0_unreweighted/init) + np.exp(p1_unreweighted/init))
reweighted_confidence[:, 0] = 100 * np.exp(p0_reweighted) / (np.exp(p0_reweighted) + np.exp(p1_reweighted))

for n in xrange(init, numSeg):
    numIter += 10
    hcrfc_un = pickle.load(open(folder + 'hcrfc_un_%d_newdata_%d_iter' % (n-init+1, numIter)))
    hcrfc_hl = pickle.load(open(folder + 'hcrfc_hl_%d_newdata_%d_iter' % (n-init+1, numIter)))

    weights_hl, dists_hl = dict2mix(hcrfc_hl.params)
    weights_un, dists_un = dict2mix(hcrfc_un.params)

    p0 = np.array([loglike_array(data_test[i, n - init + 1:n + 1, :], weights_hl, dists_hl) for i in xrange(numTestSeq)])
    p1 = np.array([loglike_array(data_test[i, n - init + 1:n + 1, :], weights_un, dists_un) for i in xrange(numTestSeq)])

    p0_unreweighted = np.array([all_loglike(data_test[i, n - init + 1:n + 1, :], weights_hl, dists_hl) for i in xrange(numTestSeq)])
    p1_unreweighted = np.array([all_loglike(data_test[i, n - init + 1:n + 1, :], weights_un, dists_un) for i in xrange(numTestSeq)])

    reweights = np.array([prob_weights[i, n - init + 1:n + 1] / np.sum(prob_weights[i, n - init + 1:n + 1]) for i in xrange(numTestSeq)])
    p0_reweighted = np.sum([p0[i] * reweights[i] for i in xrange(numTestSeq)], axis=1)
    p1_reweighted = np.sum([p1[i] * reweights[i] for i in xrange(numTestSeq)], axis=1)

    reweighted_confidence[:, n - init + 1] = 100 * np.exp(p0_reweighted) / (np.exp(p0_reweighted) + np.exp(p1_reweighted))
    unreweighted_confidence[:, n - init + 1] = 100 * np.exp(p0_unreweighted/init) / (np.exp(p0_unreweighted/init) + np.exp(p1_unreweighted/init))

np.save(folder+'reweighted_confidence', reweighted_confidence)
# np.save(folder+'unreweighted_confidence_new_test', unreweighted_confidence)
