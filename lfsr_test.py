import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def lfsr2(seed, taps, nbits):
    sr = seed
    while 1:
        xor = 1
        for t in taps:
            if (sr & (1<<(t-1))) != 0:
                xor ^= 1
        sr = (xor << nbits-1) + (sr >> 1)
        yield xor, sr
        if sr == seed:
            break
# ---------------------------- main program ---------------------------------------------
# nbits, tapindex, seed = 12, (12,11,10,4,1), 0b11001001
nbits, tapindex, seed = 8, (8,6,5,4,1), 0b00001001
datalist, movelist = [], []

# -------------------------- generate the random sequence --------------------------------

for xor, sr in lfsr2(0b11001001, tapindex, nbits):
    lfsr_gen = int(bin(2**nbits+sr)[3:], base=2)
    datalist.append(lfsr_gen)
    print (xor, lfsr_gen)


for i in datalist:
    move = i % 3        # use a mod 3 to create 3 bins
    movelist.append(move)

print ('Player1 RPS LRSR ditribution')
print ('Player1 rock:', movelist.count(0))  # count the num of ones in list (rock)
print ('Player1 paper:', movelist.count(1)) # count the num of twos in list (paper)
print ('Player1 sessios:', movelist.count(2))   # count the num of threes in list (sessiors)
print ('total moves:', len(movelist))
print ('sample moves', movelist[:20])

#---------------- print the PDF chart --------------------------------

x = np.array(datalist)
nbins = 20
n, bins = np.histogram(x, nbins, density=1)
pdfx = np.zeros(n.size)
pdfy = np.zeros(n.size)
for k in range(n.size):
    pdfx[k] = 0.5*(bins[k]+bins[k+1])
    pdfy[k] = n[k]
plt.plot(pdfx, pdfy)        # plot the probability distributed function
plt.show(block = False)
