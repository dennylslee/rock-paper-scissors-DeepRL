import numpy as np
import random
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque
from randmove import genOneMove

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# -------------------------- SETTING UP THE ENVIRONMENT --------------------------------------
# simple game, therefore we are not using the open gym custom set up
#---------------------------------------------------------------------------------------------
class RPSenv():
	def __init__ (self):
		self.action_space = [0,1,2]		# integer representation of r/p/s
		self.seed = random.seed(4) 		# make it deterministic
		self.norm_mu = 0				# center point for guassian distribution
		self.norm_sigma = 2.0			# sigma for std distribution 
		self.seqIndex = 0				# index for pointing to the SEQ sequnce 
		self.p2Mode = 'PRNG'  			# SEQ or PRNG or LFSR
		self.p2Count = [0, 0, 0] 		# player 2 win tie lost count
		self.p1Count = [0, 0, 0]		# player 1 win tie lost count
		self.window = 10					# window size for rate trending calc
		self.cumWinRate, self.cumTieRate, self.cumLostRate = None, None, None
		self.cumWinCount, self.cumTieCount, self.cumLostCount = None, None, None
		self.winRateTrend, self.tieRateTrend, self.lostRateTrend = 0, 0, 0
		self.winRateMovingAvg, self.tieRateMovingAvg, self.lostRateMovingAvg = 0, 0, 0
		self.winRateBuf, self.tieRateBuf, self.lostRateBuf \
			= deque(maxlen=self.window), deque(maxlen=self.window), deque(maxlen=self.window)
		# put all the observation state in here; shape in Keras input format
		self.state = np.array([[ \
			None, None, None, \
			self.winRateTrend, self.tieRateTrend, self.lostRateTrend, \
			self.winRateMovingAvg, self.tieRateMovingAvg, self.lostRateMovingAvg \
			]])  

	def reset(self):
		# reset all the state
		self.cumWinRate, self.cumTieRate, self.cumLostRate = 0, 0, 0
		self.cumWinCount, self.cumTieCount, self.cumLostCount = 0, 0, 0
		self.winRateTrend, self.tieRateTrend, self.lostRateTrend = 0, 0, 0
		self.winRateMovingAvg, self.tieRateMovingAvg, self.lostRateMovingAvg = 0, 0, 0
		return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

	def step(self, action, moveCount, stage):	
		# value mode is PRNG or SEQ
		p2Move = genOneMove(self, self.p2Mode, stage)			# play one move from player2
		self.p2Count[p2Move] += 1
		p1Move = action
		self.p1Count[p1Move] += 1

		# check who won, set flag and assign reward 
		win, tie, lost = 0, 0, 0
		if p1Move == p2Move:
			self.cumTieCount, tie   = self.cumTieCount  + 1, 1
		elif (p1Move - p2Move == 1) or (p1Move - p2Move == -2):
			self.cumWinCount, win   = self.cumWinCount  + 1, 1
		else:
			self.cumLostCount, lost = self.cumLostCount + 1, 1

		# update the running rates 
		self.cumWinRate = self.cumWinCount / moveCount
		self.cumTieRate = self.cumTieCount / moveCount
		self.cumLostRate = self.cumLostCount / moveCount
		# update moving avg buffer
		self.winRateBuf.append(self.cumWinRate) 
		self.tieRateBuf.append(self.cumTieRate)
		self.lostRateBuf.append(self.cumLostRate)
		# calculate trend
		tmp = [0, 0, 0]
		self.winRateTrend, self.tieRateTrend, self.lostRateTrend = 0, 0, 0
		if moveCount >= self.window:
			tmp[0] = sum(self.winRateBuf[i] for i in range(self.window)) / self.window
			tmp[1] = sum(self.tieRateBuf[i] for i in range(self.window)) / self.window
			tmp[2] = sum(self.lostRateBuf[i] for i in range(self.window)) / self.window
			# win rate trend analysis
			if self.winRateMovingAvg  < tmp[0]: 
				self.winRateTrend = 1		# win rate trending up. That's good
			else: 
				self.winRateTrend = 0		# win rate trending down. That's bad
			# tie rate trend analysis
			if self.tieRateMovingAvg  < tmp[1]:
				self.tieRateTrend = 1  		# tie rate trending up. That's bad
			else:
				self.tieRateTrend = 0  		# tie rate trending down.  Neutral
			# lost rate trend analysis
			if self.lostRateMovingAvg  < tmp[2]:
				self.lostRateTrend = 1  	# lst rate trending up.  That's bad
			else:
				self.lostRateTrend = 0  	# lost rate trending down. That's good
			self.winRateMovingAvg, self.tieRateMovingAvg, self.lostRateMovingAvg = tmp[0], tmp[1], tmp[2]
		# net reward in this round
		reward = win 									
		# record the state and reshape it for Keras input format
		dim = self.state.shape[1]
		self.state = np.array([\
			win, tie, lost, \
			self.winRateTrend, self.tieRateTrend, self.lostRateTrend, \
			self.winRateMovingAvg, self.tieRateMovingAvg, self.lostRateMovingAvg \
			]).reshape(1, dim)
		# this game is done when it hits this goal
		done = False 
		return self.state, reward, done, dim

# ------------------------- class for the Double-DQN agent ---------------------------------
# facilities utilized here:
# 1)  Double DQN networks: one for behavior policy, one for target policy
# 2)  Learn from  sample from pool of memories 
# 3)  Basic TD-Learning stuff:  learning rate,  gamma for discounting future rewards
# 4)  Use of epsilon-greedy policy for controlling exploration vs exploitation
#-------------------------------------------------------------------------------------------
class DDQN:
    def __init__(self, env):
        self.env     = env
        # initialize the memory and auto drop when memory exceeds maxlen
        # this controls how far out in history the "expeience replay" can select from
        self.memory  = deque(maxlen=2000)   
        # future reward discount rate of the max Q of next state
        self.gamma = 0.9 			  
        # epsilon denotes the fraction of time dedicated to exploration (as oppse to exploitation)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9910
        # model learning rate (use in backprop SGD process)
        self.learning_rate = 0.005	
        # transfer learning proportion contrl between the target and action/behavioral NN
        self.tau = .125 			
        # create two models for double-DQN implementation
        self.model        = self.create_model()
        self.target_model = self.create_model()
        # some space to collect TD target for instrumentaion
        self.TDtargetdelta, self.TDtarget = [], []
        self.Qmax =[]

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.state.shape[1]
        model.add(Dense(24, input_dim=state_shape, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
		# let the output be the predicted target value.  NOTE: do not use activation to squash it!
        model.add(Dense(len(self.env.action_space)))  
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        print(model.summary())

        return model

    def act(self, state):
    	# this is to take one action
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # decide to take a random exploration or make a policy-based action (thru NN prediction)
        if np.random.random() < self.epsilon:
        	# return a random move from action space
        	return random.choice(self.env.action_space)
        else:
        	# return a policy move
        	self.Qmax.append(max(self.model.predict(state)[0]))
        	return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
		# store up a big pool of memory
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):  		# DeepMind "experience replay" method
    	# the sample size from memory to learn from
        batch_size = 32
        # do nothing untl the memory is large enough
        if len(self.memory) < batch_size: return
        # get the samples
        samples = random.sample(self.memory, batch_size)
        # do the training (learning); this is DeepMind tricks of using "Double" model (Mnih 2015)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            #print('target at state is ', target)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0]) 
                TDtarget = reward + Q_future * self.gamma
                self.TDtarget.append(TDtarget)
                self.TDtargetdelta.append(TDtarget - target[0][action])
                target[0][action] = TDtarget	 			
            # do one pass gradient descend using target as 'label' to train the action model
            self.model.fit(state, target, epochs=1, verbose=0)
        
    def target_train(self):
    	# transfer weights  proportionally from the action/behave model to the target model
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

# ------------------------- MAIN BODY ----------------------------------------

def main():
	episodes, trial_len =  150, 300					# lenght of game play
	stage, totalStages = 0, 5						# # of stages with change distribution
	sigma_reduce = -0.1								# sigma change amount at each stage
	cumReward, argmax = 0, 0						# init for intrumentation
	steps, rateTrack = [], []
	avgQmaxList, avgQ_futureList,avgQ_targetmaxList, avgTDtargetList = [], [], [], []
	avgCumRewardList = []
	p1Rate, p2Rate = [], []
	# declare the game play environment and AI agent
	env = RPSenv()
	dqn_agent = DDQN(env=env)
	# ------------------------------------------ start the game -----------------------------------------
	print('STARTING THE GAME with %s episodes each with %s moves' % (episodes, trial_len), '\n')
	for episode in range(episodes):
		cur_state = env.reset().reshape(1,env.state.shape[1])   # reset and get initial state in Keras shape
		cumReward = 0
		# ----------------- Select play strategy (apply to PRNG mode only) ------------------------------
		#   this change strategy affects distribution of r-p-s
		# -----------------------------------------------------------------------------------------------
		if (episode+1) % (episodes // totalStages) == 0: env.norm_sigma += sigma_reduce	# step repsonse change the gaussian distribution
		#   this change strategy affects which move is dominant (control in randmove module)
		stage = episode // (episodes // totalStages)									# divide total episodes into 3 equal length stages
		
		for step in range(trial_len):
			# AI agent take one action
			action = dqn_agent.act(cur_state)
			# play the one move and see how the environment reacts to it
			new_state, reward, done, info = env.step(action, step + 1, stage)
			cumReward += reward
			# record the play into memory pool
			dqn_agent.remember(cur_state, action, reward, new_state, done)
			# perform Q-learning from using |"experience replay": learn from random samples in memory
			dqn_agent.replay()
            # apply tranfer learning from actions model to the target model.
			dqn_agent.target_train() 
			# update the current state with environment new state
			cur_state = new_state
			if done:  break
		#-------------------------------- INSTRUMENTAL AND PLOTTING -------------------------------------------
		# the instrumental are performed at the end of each episode
		# store epsiode #, winr rate, tie rate, lost rate, etc. etc.
		#------------------------------------------------------------------------------------------------------
		rateTrack.append([episode+1, env.cumWinRate, env.cumTieRate, env.cumLostRate])
		if True:		# print ongoing performance
			print('EPISODE ', episode + 1), 
			if env.p2Mode == 'PRNG': print('stage:', stage, ' sigma:', env.norm_sigma)
			print(' WIN RATE %.2f ' % env.cumWinRate, \
				'    tie rate %.2f' % env.cumTieRate, \
				'lose rate %.2f' % env.cumLostRate)
		
		# print move distribution between the players
		if True:
			p1Rate.append([env.p1Count[0] / trial_len, env.p1Count[1] / trial_len, env.p1Count[2] / trial_len])
			p2Rate.append([env.p2Count[0] / trial_len, env.p2Count[1] / trial_len, env.p2Count[2] / trial_len])
			print (' P1 rock rate: %.2f paper rate: %.2f scissors rate: %.2f' %  (p1Rate[-1][0], p1Rate[-1][1], p1Rate[-1][2]))
			print (' P2 rock rate: %.2f paper rate: %.2f scissors rate: %.2f' %  (p2Rate[-1][0], p2Rate[-1][1], p2Rate[-1][2]))
			env.p1Count, env.p2Count = [0,0,0], [0,0,0]
		
		# summarize Qmax from action model and reward 
		avgQmax = sum(dqn_agent.Qmax) / trial_len  	# from action model
		avgQmaxList.append(avgQmax)

		avgCumReward = cumReward / trial_len
		avgCumRewardList.append(avgCumReward)
		if True:
			print(' Avg reward: %.2f Avg Qmax: %.2f' % (avgCumReward, avgQmax))
		dqn_agent.Qmax=[] 		# reset for next episode


	# ---------------- plot the main plot when all the episodes are done ---------------------------
	#
	if True:
		fig = plt.figure(figsize=(12,5))	
		plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
		
		# plot the average Qmax
		rpsplot = fig.add_subplot(321)
		plt.title('Average Qmax from action model', loc='Left', weight='bold', color='Black', \
			fontdict = {'fontsize' : 10})
		rpsplot.plot(avgQmaxList, color='blue')
		
		# plot the TDtarget
		rpsplot = fig.add_subplot(323)
		plt.title('TD target minus Q target from experience replay', loc='Left', weight='bold', \
			color='Black', fontdict = {'fontsize' : 10})
		rpsplot.plot(dqn_agent.TDtarget, color='blue')
		
		# plot the TDtarget
		rpsplot = fig.add_subplot(325)
		plt.title('TD target from experience replay', loc='Left', weight='bold', color='Black', \
			fontdict = {'fontsize' : 10})
		rpsplot.plot(dqn_agent.TDtargetdelta, color='blue')
		
		# plot thte win rate
		rpsplot = fig.add_subplot(322)
		plt.title('Win-Tie-Lost Rate', loc='Left', weight='bold', color='Black', \
			fontdict = {'fontsize' : 10})
		rpsplot.plot([i[1] for i in rateTrack], color='green')
		rpsplot.plot([i[2] for i in rateTrack], color='blue')
		rpsplot.plot([i[3] for i in rateTrack], color='red')
		
		# plot thte win rate
		rpsplot = fig.add_subplot(324)
		plt.title('Player 2 move percentage', loc='Left', weight='bold', color='Black', \
			fontdict = {'fontsize' : 10})
		rpsplot.plot([i[0] for i in p2Rate], color='orange')
		rpsplot.plot([i[1] for i in p2Rate], color='red')
		rpsplot.plot([i[2] for i in p2Rate], color='green')
		
		# plot the reward 
		rpsplot = fig.add_subplot(326)
		plt.title('Average Reward per Episode', loc='Left', weight='bold', color='Black', \
			fontdict = {'fontsize' : 10})
		rpsplot.plot(avgCumRewardList, color='green')
		plt.show(block = False)
	
if __name__ == "__main__":
	main()
