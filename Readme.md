# Introduction

The objective of this project is to construct an AI agent to play a simple 2 players rock-paper-scissors game using reinforcement learning (RL) technique - particularly the double-DQN algorithm.  

In previous design project, we have built a player using simple LSTM-based neural network that is trained using traditional supervised learning method.  The downside of that is the model is effectively static and does not adapt to model drift or fundamental behavioral changes.  With a RL-based approach, we like to see that the AI agent demostrate some ability to counter the changing strategy of the opponent and continue to generate better win rate than the opposing player. 

## Q-Learning Basics

This project utiltized the double-DQN RL algorithm as the basis for the AI agent as player 1.  DeepMind was very instrutmental in pinoeering and popularizing Q-learning with multple enhancements - thought Q-Learning was first introduced by Watkins in 1992.  See few popular papers on DQN (deep Q-learning network) in the reference section.

In its essence, Deep Q-Learning is to learn a "policy" (which in practice is a deep learning network) such that it maximizes the expected return of future action. The Q-value is the "action value" function. The Q-value's absolute magnitude has no real meaning but is recurrsively derived to guide the system which action to take at the moment such that such action is believed to lead to maximum future rewards. 

Some of the enhancments utilized in this design over beyond the basic Q-learning (which are introduced mainly be DeepMind in recent years) are:

1) Dual models (hence the term "double") - one to drive the action decision (also sometime refers as online model, behavior model) and one to act as the target model.
2) Exploiting vs exploration using epsilon greedy method. This is essentia in any RL design since exploration is important, especially during the start of the process, to search in different areas of state space which might lead to better optimal operation position.  When exploring, it is often termed "off-policy" whereas taking an action according to exploitatin is often termed "on-policy".
3) Experience reply - instead of using most recent history as the learning space, DeepMind introduced the concept of experience reply in which past pass through of the state space is stored in memory.  Such memory are recalled (sampled) at each move and used in the SGD process (thus achieving the reinforcement notion)

## Acknowledgement

Much of the code is adopted from A. Oppermann's blog in Medium. It is an excellent tutorial and walk through. You can find it [here](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47)

# RPS High Level Environment 

![pic1](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/high_level_play_environment.png)


# RPS AI player architecture using Double-DQN

![pic2](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/double_dqn_architecture.png)


## Player 2 - "the opponent"


## Hyperparameters

## Results



![pic3](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/main_results_plot.png)

## Future Works 

## Reference

1. Mnih, et.al.  "Playing Atari with Deep Reinforcement Learning", 2013
2. Hasselt, et.al. "Deep Reinforcement Learning with Dobule Q-learning", 2015
3. Mnih, et. al. "Human-Level control through deep reinforcement learning"
4. Packer, et. al. "Assessing Generalization in Deep Reinforcement Learning"
5. A. Oppermann, "self learing AI agents part-II Deep Q-learning", 2018

