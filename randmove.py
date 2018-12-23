# This module generate one rock-paper-sessiors move based on the Mode selected.
import random

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
	else:
		print('Error: random mode does not exist!')

