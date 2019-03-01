# Introduction

The objective of this project is to construct an AI agent to play a simple 2 players rock-paper-scissors game using reinforcement learning (RL) technique - particularly the double-DQN algorithm.  

In previous design project, we have built a player using simple LSTM-based neural network that is trained using traditional supervised learning method.  The downside is that the model is effectively static and does not adapt to model drift or fundamental behavioral changes.  With a RL-based approach, we like to see that the AI agent demostrates some ability to counter the changing strategy of the opponent and continue to generate better win rate than the opposing player.  

Note that unlike other RL project in which the AI agent is to learn a task successfully and generalizes to other future data (i.e. a form of semi supervised learning using rewards as label - this is my analogy and is technically not precise), in this particular set up, there is no completion per se.  That is the game never ends. The RL agent simply continues to adapt and adjust as best as it can. 

## Q-Learning Basics

This project utiltized the double-DQN RL algorithm as the basis for the AI agent as player 1.  DeepMind was very instrutmental in popularizing Q-learning recently and had contributed multple architectural enhancements in recent years - do note that Q-Learning was first pioneered by Watkins in 1992.  See few popular papers on DQN (deep Q-learning network) in the reference section.

In its essence, Deep Q-Learning is to learn a "policy" (which in practice is a deep learning network) such that it maximizes the expected return of future action. The Q-value is the "action value" function. Its absolute magnitude has no real meaning but is recurrsively derived to guide the system on which action to take at any moment such that the chosen action is believed to lead to maximum future rewards. 

Some of the enhancments utilized in this design that is over and beyond basic Q-learning are as following.  Most of these are introduced mainly by DeepMind in recent years. 

1) Dual models (hence the term "double") - one model is to drive the action decision (also sometime refers as online model or behavior model) and one to act as the target model.  The inner weights are "tranferred" from the action model to the target model on every move cycle on a discounted basis.
2) Exploiting vs exploration using epsilon-greedy method. This is essential in any RL design since exploration is important, especially during the start of the process, for searching in different areas of that state space that might lead to better optimal operation position.  When exploring, it is often termed "off-policy" whereas taking an action according to exploitation is often termed "on-policy".
3) Experience replay - instead of using most recent history as the learning space, DeepMind introduced the concept of experience replay in which past pass-through of the state space is stored in memory.  Such memories are recalled (sampled) at each move and used in the SGD process (thus achieving the reinforcement notion).

## Acknowledgement

Much of the code is adopted from A. Oppermann's blog in Medium. It is an excellent tutorial with detailed walk through. You can find it [here](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47).

# RPS High Level Environment 

The environment of this game play is depicted below. It follows the classical RL environment definition.

![pic1](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/high_level_play_environment.png)

Inside the "environment" is the embedded player 2 (i.e. the opponent).  This player might adopt different type of play strategy. The interaction contains the following:

1. action space:  either rock, paper, or scissors that the AI agent (player 1) puts out.
2. rewards:  this is an indication from the environment back to player 1.  The reward is simply a value of 1 if it is a win for player 1, and 0 otherwise.
3. state:  this is where the fun is and some creativity comes into play (which might affect the player winning outcome). In this setup, we have designed the state space to be:
    - win, tie, lost indicators: one of the three can be set to a 1 for a particular state
    - winRateTrend, tieRateTrend, lostRateTrend:  this is an indicator which reflects a positively-trending moving average (set to 1) or not (set to 0).  All three indicators are assessed independently.
    - winRateMovingAvg, tieRateMovingAvg, lostRateMovingAvg: floating point value between 0 and 1 which indicates the rate. This rate is calculated based on a configured moving average window size.


## RPS AI player architecture using Double-DQN

The overall architecture of the double-DQN agent design is depicted below.  The yellow section is coded inside the step method and is iterated over experience replay batch size of N samples.  The green section is the design that control the exploration vs exploitation action.  The oragne section is the main action model (on policy).  This is the model in which we want to ultimately achieve the optimal policy for making the best possible action.  The action model's weights are transferred with discounted to the target model periodically. 

![pic2](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/double_dqn_architecture.png)

(Nothing really special here in the illustration - just a block diagram version of the commonly published Dobule-DQN pesudo code)

## Player 2 ("the opponent") behaviors

The player 2 code is embedded in a separate moduled called randmove. Player 2 can play 3 modes which need to be manually configured in the environment class setup:  

1. "PRNG" which uses python Gaussian distribution with a controllable mean and sigma value.  The larget the sigma, the larger the spread and the more random the sequence appears (which makes it harder to predict).
2. "SEQ" this is a simple manually typed in sequence of rock-paper-scissors.
3. "LFSR"  this is a N-bits pseudo-random generator implemented based on linear feedback shift register of certain length.  Depending on the tap location, it might or might exhibit maximal length cycle. 

The results below is based mainly on PRNG in which the play strategy is as following:

1) the overall game play is divided into N (=5) stages, each with equal number of moves
2) the moves of player 2 in each stage is generated using the PRNG Guassian distribution with different dominant move type (i.e. rock or paper of scissors)
3) the sigma value is also decreasesd as the stage progress.

The overall game objective is to observe if the RL agent can adjust to the changing behaviors and maintain a good win rate.
 
## DNN design

In this project, we used a simple full mesh DNN which proves to be one of the main limitation since the RPS game is essentially a time series (sequence) problem. The DNN model can learn distribution but it has no notion of sequence. 

For a high dimensional PRNG like the Guassian generator, a DNN (especially a small one) has no hope in detecting the sequential pattern coming from a complex PRNG generator.  However, it is capable in understanding the distribution statistically and its play strategy is effectively based on observing shift in statistics. 

## Hyperparameters

The following is the set of hyperparammeters in the game which can alter the outcome win rate and in general the adaptability of the AI agent.

1. memory batch size = 32: the larger batch, the more reinforcement is used on each round making the convergence faster but would also make adaptation slower.
2. memory maxlen: this is done using python collection deque object. The longer the memory, the more experience it retains which could lead to faster convergence but can negatively affect adaptability since it needs time to flush out the memory when the opponents' behavior changes. 
3. moving window size = 8: change the smoothing factor of moving average state variables.
4. gamma - a factor to determine how much of the future action-value do you want the algorithm to incorporate at each step.
5. DNN layers = 3:  effectively the entropy of the NN and its ability to learning the distribution
6. DNN nodes = 64: same as above
action to target model transfer learning (tau): how fast the transfer mimic after the action model. (we didn't play much around this one)
6. reward system = 1 for win, 0 otherwise: using simple reward and allows the complexity of adaptation controlled by the NN design and the state space design.
7. state design = too much state indicatores is a waste of computation resource since not all state variables are important.  Too little would hinder what the RL agent observes thus limits its policy forming ability. 
8. epsilon: exploration percentage controlled in conjunction with the decay rate.  Note that there is a minimum value which is important to the adaptative effective since the system will continue to explore off-policy moves and it is how it discovers new opponent's behavior. 

# Results

The player 2's move percentage clearly depictes the changing strategy across the different stages of the game - each stage with a different dominant move type and a different spread due to the decreasing sigma value.  Player 1 (RL agent) seems to successfully adapt to the change and eventually win more due to the weakened player 2 behaviors (i.e. it gets less random over time).  From that perspective, the code has successfully achieved its main objective. 

However, it is somewhat disappointing that win rate in any state does not outperfrom the statistical behaviors of its opponent since the DNN architecture has the inherent limitation as mentioned in the earlier section. 

![pic3](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/main_results_plot.png)

The max Q value of action model is also plotted.  It shows a nice convergenece over the length of game. Corresponding the Q-value is upsetted during the transition from one stage to another when the opponent changes its play strategy.  During that time, reward is not received based given the previous policy's strategy and corresponding the Q-value suffered.  However, the exploration allows new policy to be learned by building up new memories and the new winning counter-move bubbles back up to the top eventually and the winning rate recuperates. 

# Using LSTM as policy and target network

As a second phase to this project, I have evolved the design to utilize a LSTM-based RNN in both the policy (action and target) networks.  The architecture is rather brute force and is illustrated below.  The intuition is that LSTM should over time learn to perform better than a simple DNN given its inert ability to recognize sequential pattern.  In a nutshell, the DDQN's original DNN model (for both action and target model) is swapped with a LSTM.  Since the LSTM deals with, and is trained on, sequence; the overall data preparation design is changed to adjust for this arrangeement:
1) the input to the LSTM is a sequence of states. The sequence is of length 'lookback'
2) the DDQN experience replay concept is still retained.  However, each experience is now a sequence of states of lenght lookback.  For example, each experience replay is a sample from the deque memory, the code then retrieves the immediate prior loopback number of states.  The code then repeat the same retrieval process RL_batch number of times. 

![pic4](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/LSTM_based_ddqn_architecture.png)

In general,  such change in architecture did not yield any significant breakthrough in results (and to some degree, it is worse performing than a simple claissically trained static LSTM model).

1) It did not perform well (meaning win-tie-lost rate are roughly 33% each) when the opponent is 'PRNG'.  The opponent is simply too high dimensions. 
2) Some testing were done aginst a simplier 12bit 'LFSR' which means the sequence repeats itself in 4,096 moves.  
- GRU vs LSTM were used with no apparent difference
- the hyperparameters varied are (a) lookback length (b) inner LSTM unit size (b) experience reply batch size.
- a flatten dense layer architecture versus a basic many-to-one LSTM architecture were tried and neither provide any apparent advantage (neither one improveds the performance)
- see the captured results below.
3)  Further testing was conducted on a short (approx 30moves) self-entered r-p-s sequence.  Based on a fairly small size LSTM, a consistently higher win rate is observed.  But this does not surpass the performance a classical (supervised learning) approach using a statically-trained LSTM.

All-in-all this LSTM architecture working within a DDQN structure is rather non-performing and better design is desirable. 

![pic5](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/figure_LSTM_lkbk200_unit100_1stage.png)

![pic6](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/figure_LSTM_lkbk100_unit50_SEQ1.png)

# Future Works 

We will need to rethink a more appropropriate LSTM-based solution to improve the win rate performance.

# Reference

1. Mnih, et.al.  "Playing Atari with Deep Reinforcement Learning", 2013
2. Hasselt, et.al. "Deep Reinforcement Learning with Dobule Q-learning", 2015
3. Mnih, et. al. "Human-Level control through deep reinforcement learning", 2015
4. Packer, et. al. "Assessing Generalization in Deep Reinforcement Learning"
5. A. Oppermann, "self learing AI agents part-II Deep Q-learning", 2018
6. Watkins & Dayan, "Q-Learning", 1992

