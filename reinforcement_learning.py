# -*- coding: utf-8 -*-
"""Reinforcement Learning.ipynb

**Frozen Lake using the Opengym AI environment**

Frozen lake is a traditional game with can be implemented in the Opengym AI environment. Basically the aim of the agent is to reach the goal from the starting point by avoiding the holes or traps.

In this assignment the Frozen lake-V1 is implemented using two algorithms which are:



1.   Q-learning Algorithm
2.   Monte Carlo method

# Q-Learning Algorithm
"""

#This python file aims at using a basic Q-Learning implementation to train an agent in the FrozenLake-v1 environment of gym

import numpy as np
import matplotlib.pyplot as plt
import gym

#Creating a class for the agent

class Agent(object):
	def __init__(self,env,gamma=0.98,alpha=0.7):

		self.env=env
		self.state_size=env.observation_space.n							        #Number of possible states/configurations
		self.action_size=env.action_space.n								          #Number of possible actions to take
		self.gamma=gamma												                    #Discount factor to take future rewards into acount
		self.alpha=alpha												                    #Learning rate, to update the Q-table
		self.action_space=np.arange(self.action_size)					      #Defining the array of the possible actions the agent can take
		self.qtable=np.zeros((self.state_size,self.action_size))		#Initializing the Q-Table
		self.reward_history=[]											                #To maintain track of average reward per episode while training
		self.episode_lengths=[]											                #To keep track of the length od episodes

	def epsilon_greedy(self,state,epsilon=0.2):

		qvalues=self.qtable[state,:]
		A=np.zeros((self.action_size)) + epsilon/self.action_size
		greedy_action=np.argmax(qvalues)
		A[greedy_action]+=1-epsilon
		return np.random.choice(self.action_space,p=A)

	#Updating the qtable for each new state vsited (for the first time) in each episode
	def update_qtable(self,state,action,reward,next_state):

		next_action=int(np.argmax(self.qtable[next_state,:]))
		self.qtable[state,action]+=self.alpha*(reward+self.gamma*self.qtable[next_state,next_action]-self.qtable[state,action])

	#Defining how a single episode would roll out
	def qlearning_episode(self):
		episode=[]
		state_now=self.env.reset()
		while True:
			action=self.epsilon_greedy(state_now)
			next_state,reward,done,_=self.env.step(action)
			episode.append([state_now,action,reward,next_state])
			state_now=next_state
			if done==True:
				break
		return np.array(episode)

	def train(self,num_episodes=10000):

		#Iterating over a number of episodes
		for i in range(num_episodes):
			#Generating the episode
			episode=self.qlearning_episode()
			#Keeping track of the length of the episode
			self.episode_lengths.append(len(episode[:,0]))
			#Updating the particular qvalues only
			for k in range(len(episode[:,0])):
				self.update_qtable(int(episode[k,0]),int(episode[k,1]),float(episode[k,2]),int(episode[k,3]))
			self.reward_history.append(np.mean(episode[:,2]))
			if (i+1)%100==0:
				print("Average reward claimed by the agent in episode {} : {}".format(i+1,self.reward_history[-1]))
				print("Length of episode {} : {}".format(i+1,self.episode_lengths[-1]))


env=gym.make("FrozenLake-v1")

agent=Agent(env=env)
agent.train()

reward_history_q=agent.reward_history

#Plotting the averge rewards and episode Lengths gained throughout each episode per episode
fig, axs = plt.subplots(1,2)
axs[0].plot(agent.reward_history)
axs[0].set_title('Average Reward per Episode')
axs[1].plot(agent.episode_lengths, 'tab:orange')
axs[1].set_title('Episode_Length')

plt.show()

print('Average Reward throughout all episodes={}'.format(sum(reward_history_q)/len(reward_history_q)))

"""# Monte Carlo Method"""

#This python file aims at using a basic Monte Carlo implementation to train an agent in the FrozenLake-v1 environment of gym

import numpy as np
import matplotlib.pyplot as plt
import gym


class Agent(object):
	def __init__(self,env,gamma=0.98,alpha=0.5):

		self.env=env
		self.state_size=env.observation_space.n							                 #Number of possible states/configurations
		self.action_size=env.action_space.n								                   #Number of possible actions to take
		self.gamma=gamma												                             #Discount factor to take future rewards into acount
		self.alpha=alpha												                             #Learning rate, to update the Q-table
		self.action_space=np.arange(self.action_size)					               #Defining the array of the possible actions the agent can take
		self.qtable=np.zeros((self.state_size,self.action_size))		         #Initializing the Q-Table
		self.reward_history=[]											                         #To maintain track of average reward per episode while training
		self.episode_lengths=[]											                         #To keep track of the length od episodes

	def epsilon_greedy(self,state,epsilon=0.2):

		qvalues=self.qtable[state,:]
		A=np.zeros((self.action_size)) + epsilon/self.action_size
		greedy_action=np.argmax(qvalues)
		A[greedy_action]+=1-epsilon
		return np.random.choice(self.action_space,p=A)

	def discounted_rewards(self,rewards):

		#Rewards is a 1-D array with the stored rewards obtained at each timestep of an episode
		current_reward=0
		discounted_rewards=np.zeros((len(rewards)))
		for t in reversed(range(len(rewards))):
			current_reward = self.gamma*current_reward + rewards[t]
			discounted_rewards[t]=current_reward
		return discounted_rewards

	def update_qtable(self,state,action,reward_discounted):

		self.qtable[state,action]+=self.alpha*(reward_discounted-self.qtable[state,action])

	def monte_carlo_episode(self):

		#Initialising the episode buffer to store the current state
		episode=[]
		#Starting from the initial starting point for the beginning of each episode
		state_now=self.env.reset()
		while True:
			#Choosing the action as per the constant epsioln greedy policy
			action=self.epsilon_greedy(state_now)
			state_next,reward,done,_=self.env.step(action)
			episode.append([state_now,action,reward])
			state_now=state_next
			if done==True:
				break
		return np.array(episode)

	def train(self,num_episodes=10000):

		#Iterating over a number of episodes
		for i in range(num_episodes):
			#Generating the episode
			episode=self.monte_carlo_episode()
			#Keeping track of the length of the episode
			self.episode_lengths.append(len(episode[:,0]))
			#Storing the reards for the episode, and discounting it
			rewards=episode[:,2].copy()
			rewards=self.discounted_rewards(rewards)
			#Updating the particular qvalues only
			for k in range(len(episode[:,0])):
				self.update_qtable(int(episode[k,0]),int(episode[k,1]),float(rewards[k]))
			self.reward_history.append(np.mean(episode[:,2]))
			if (i+1)%100==0:
				print("Average reward claimed by the agent in episode {} : {}".format(i+1,self.reward_history[-1]))
				print("Length of episode {} : {}".format(i+1,self.episode_lengths[-1]))


env=gym.make("FrozenLake-v1")

agent=Agent(env=env)
agent.train()

reward_history_monte=agent.reward_history

#Plotting the averge rewards and episode Lengths gained throughout each episode per episode
fig, axs = plt.subplots(1,2)
axs[0].plot(agent.reward_history)
axs[0].set_title('Average Reward per Episode')
axs[1].plot(agent.episode_lengths, 'tab:orange')
axs[1].set_title('Episode_Length')

plt.show()

print('Average Reward throughout all episodes={}'.format(sum(reward_history_monte)/len(reward_history_monte)))

Average_q=sum(reward_history_q)/len(reward_history_q)
Average_monte=sum(reward_history_monte)/len(reward_history_monte)

print('Average Reward for Q-Learning={}'.format(Average_q))
print('Average Reward for Monte Carlo={}'.format(Average_monte))