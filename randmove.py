# This module generate one rock-paper-sessiors move based on the Mode selected.
import random

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

def genOneMove(self, mode, stage):				
	if mode == 'PRNG':
		# change play strategy for player2 along the way 
		lowPlay  = {0:0, 1:1, 2:2} 						# key = stage number, value = r(0), p(1), s(2)
		meanPlay = {0:1, 1:2, 2:0} 						# key = stage number, value = r(0), p(1), s(2)
		hiPlay   = {0:2, 1:0, 2:1} 						# key = stage number, value = r(0), p(1), s(2)
		# gen a random numbe from guassian & quantize it
		a = random.gauss(self.norm_mu, self.norm_sigma) 
		if a **2 < 1:  									# the middle bell is the paper move
			play = meanPlay[stage]
		elif a < -1:									# lower than cutoff -1 is the rock move
			play = lowPlay[stage]
		else:
			play = hiPlay[stage] 						# else higher than +1 is the sessiors move 
		return play
	
	elif mode == 'SEQ':					# simple repeating pattern as 'random generator'
		dict = {'r':0, 'p':1, 's': 2}
		seqlist = 'rrpprsspsr'			# the pattern sequence here
		self.seqIndex = 0 if self.seqIndex == len(seqlist)-1 else self.seqIndex + 1
		return dict[seqlist[self.seqIndex]]
	
	elif mode == 'LFSR':
		nbits, tapindex, seed = 12, (12,11,10,4,1), 0b11001001
		#nbits, tapindex, seed = 8, (8,6,5,4,1), 0b11001001
		lfsrlist = []
		for xor, sr in lfsr2(seed, tapindex, nbits):
		    lfsr_gen = int(bin(2**nbits+sr)[3:], base=2)
		    lfsrlist.append(lfsr_gen % 3)
		self.seqIndex = 0 if self.seqIndex == len(lfsrlist)-1 else self.seqIndex + 1
		return lfsrlist[self.seqIndex]
	
	else:
		print('Error: random mode does not exist!')

