import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import style
style.use('ggplot')

num_bins = 100
sigma1, sigma2 = 2.0, 0.5
norm_mu = 0
a, b = [], []
for i in range(10000):
	a.append(random.gauss(norm_mu, sigma1))
	b.append(random.gauss(norm_mu, sigma2))

plt.title('Distribution Plots', loc='center', weight='bold', color='Black')

#n, bins1, patches1 = plt.hist(a, num_bins, facecolor='blue', alpha=0.5)
m, bins2, patches2 = plt.hist(b, num_bins, facecolor='red', alpha=0.5)
plt.show(block = False)

# some test code
teststr = 'start'
for i in range(50):
	j = 61
	k = 2
	teststr = 'start'
	if i % (j // k) == 0: teststr = 'change'
	print ('i', i, teststr, i % (j // k))
