import matplotlib.pyplot as plt
import numpy as np


segLens = [40]
dim = 4
run = 1

# confidence = np.array([np.load('./results/time_freq_dim%d_%ds_%d-th_run/confidence.npy' % (dim, segLen/4, run)) for run in runs])
# reweighted_confidence = np.array([np.load('./results/time_freq_dim%d_%ds_%d-th_run/reweighted_confidence.npy' % (dim, segLen/4, run)) for run in runs])

pH = np.load('pH.npy')
idx = np.load('./index/idx.npy')
idx_unhealthy_test = np.array([17, 35, 117, 120, 129, 146, 179, 186, 324, 328, 405, 465, 469, 493, 548])
pH_Unhealthy_test = pH[idx_unhealthy_test]
pH_test = pH[idx]
u = pH_test[0:len(pH_test)/2][30:]
h = pH_test[len(pH_test)/2:][30:]

segLen = segLens[0]
confidence = np.load('./results/time_freq_dim%d_%ds_%d-th_run/confidence.npy' % (dim, segLen/4, run))
reweighted_confidence = np.load('./results/time_freq_dim%d_%ds_%d-th_run/reweighted_confidence.npy' % (dim, segLen/4, run))

'''
plt.subplot(221)
[plt.plot(confidence[i][4], label='%d run' % (i+1)) for i in xrange(len(runs))]
# plt.legend()
plt.ylim((40, 75))
plt.title('1st FHR sequence')

plt.subplot(222)
[plt.plot(confidence[i][1], label='%d run' % (i+1))for i in xrange(len(runs))]
# plt.legend()
plt.ylim((40, 85))
plt.title('2nd FHR sequence')

plt.subplot(223)
[plt.plot(confidence[i][7], label='%d run' % (i+1))for i in xrange(len(runs))]
# plt.legend()
plt.ylim((40, 75))
plt.title('5th FHR sequence')

plt.subplot(224)
[plt.plot(confidence[i][9], label='%d run' % (i+1))for i in xrange(len(runs))]
# plt.legend()
plt.ylim((40, 75))
plt.title('6th FHR sequence')
'''

plt.subplot(221)
[plt.plot(confidence[i], label=h[i]) for i in xrange(5)]
plt.legend()
plt.title('confidence of healthy fetuses')

plt.subplot(222)
[plt.plot(reweighted_confidence[i], label=h[i]) for i in xrange(5)]
plt.legend()
plt.title('re-weighted confidence of healthy fetuses')

plt.subplot(223)
[plt.plot(confidence[i], label=u[i-5]) for i in xrange(5, 10)]
plt.legend()
plt.title('confidence of unhealthy fetuses')

plt.subplot(224)
[plt.plot(reweighted_confidence[i], label=u[i-5]) for i in xrange(5, 10)]
plt.legend()
plt.title('re-weighted confidence of unhealthy fetuses')
'''
recordings = np.array([0, 1, 3, 8, 9])
pH_tmp = np.concatenate((h, u))

plt.subplot(321)
[plt.plot(confidence[i], label=pH_tmp[i]) for i in recordings]
plt.legend()
plt.title('probability by naive approach')

plt.subplot(322)
[plt.plot(reweighted_confidence[i], label=pH_tmp[i]) for i in recordings]
plt.legend()
plt.title('probability by weighted approach')

segLen = segLens[1]
confidence = np.load('./results/time_freq_dim%d_%ds_%d-th_run/confidence.npy' % (dim, segLen/4, run))
reweighted_confidence = np.load('./results/time_freq_dim%d_%ds_%d-th_run/reweighted_confidence.npy' % (dim, segLen/4, run))

plt.subplot(323)
[plt.plot(confidence[i], label=pH_tmp[i]) for i in recordings]
plt.legend()
plt.title('probability by naive approach')

plt.subplot(324)
[plt.plot(reweighted_confidence[i], label=pH_tmp[i]) for i in recordings]
plt.legend()
plt.title('probability by weighted approach')

segLen = segLens[2]
confidence = np.load('./results/time_freq_dim%d_%ds_%d-th_run/confidence.npy' % (dim, segLen/4, run))
reweighted_confidence = np.load('./results/time_freq_dim%d_%ds_%d-th_run/reweighted_confidence.npy' % (dim, segLen/4, run))

plt.subplot(325)
[plt.plot(confidence[i], label=pH_tmp[i]) for i in recordings]
plt.legend()
plt.title('probability by naive approach')

plt.subplot(326)
[plt.plot(reweighted_confidence[i], label=pH_tmp[i]) for i in recordings]
plt.legend()
plt.title('probability by weighted approach')
'''

plt.show()
