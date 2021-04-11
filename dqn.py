
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, concatenate, Dense, Flatten

from agent import Agent
from env import Env 


class DQN:
	def __init__(self):
		# Initialise model parameters. Episode size, intial bank will be fixed. 
		# Training data chosen to be Jan, Feb, Mar.
		# Validation data chosen to be Apr. 

		self.window_size = 10			# will be adjusted
		self.episode_size = 120 		# fixed
		self.train_episodes = 386
		#self.train_episodes  = 10
		self.batch_size = 32
		self.initial_bank = 200
		self.max_ts = 200

		# Exploitation / Exporation parameters
		self.epsilon = 1
		self.epsilon_decay = 0.99
		self.min_epsilon = 0.1

		self.val_start = 46340
		self.val_end = 60680
		#self.val_end = 60750


	def set_variables(self):
		# Initialise agent, environment and stats dataframe. 
		# Formatted in this function so we can modify NN models and input data in future 
		# experiments. 

		df = pd.read_excel('data/eur-gbp.xlsx', 
			names=['date','open','high','low','close','volume'])
		data = df['close'].to_numpy()
		self.data = data.reshape(1, len(data))	

		model = self.get_model()	
		self.agent = Agent(model, self.batch_size, self.max_ts)
		self.env = Env(self.initial_bank)


		stat_columns = {'price','ts','episode','action','inv','reward', 'bank', 'portfolio'}
		self.model_stats = pd.DataFrame(columns=stat_columns)


	def get_model(self):
		# Returns a pre-determined and fixed model for current experimentation.

		# Input Layers
		price_input = Input(shape=(self.window_size,), name='price_input')
		env_input = Input(shape=(2,), name='env_input')

		# Adjsustable Layers
		price_layer1 = Dense(32, activation='relu', name='price_layer1')(price_input)
		price_layer2 = Dense(16, activation='relu', name='price_layer2')(price_layer1)
		price_final = Flatten(name='price_flatten')(price_layer2)

		# Fixed layers
		concat_layer = concatenate([price_final, env_input], name='concat_layer')
		fixed_layer1 = Dense(8, activation='relu', name='fixed_layer1')(concat_layer)
		fixed_layer2 = Dense(4, activation='relu', name='fixed_layer2')(fixed_layer1)

		# Output Layer
		action_output = Dense(1, activation='linear', name='action_output')(fixed_layer2)

		model = Model(inputs=[price_input, env_input], outputs=[action_output])
		model.compile(optimizer='SGD', loss={'action_output':'mse'})

		return model


	def get_state(self, start):
		# Returns state/observation of current environment.
		#
		# param 	start 	Initial timestep of windowed data
		# output 	state 	State of current environment

		prices = self.data[:, start:start+self.window_size ]
		env_state = [self.env.bank, len(self.env.inventory)]
		state = [prices, env_state]
		return state


	def learn(self, episode, start, end):
		# Learning stage of DQN model. Selects and execute actions, then
		# saves parameters in replay memory to run experience replay.
		#
		# param 	episode 	Episode number 
		# 							-1 if in validation stage
		# param 	start 		Initial timestep of episode
		# param 	end 		Terminal timestep of episode

		# reset environment
		self.env.reset()

		# Current timestep
		current_state = self.get_state(start)

		done = False
		for t in range(start, end):
			if t == end-1:
				done = True

			current_price = current_state[0][0][-1]
			
			# Determie and execute action
			action = self.agent.act(current_state, self.epsilon)
			execute = self.env.executable(action, current_price)
			if not execute:
				action = self.agent.explore(current_state)
			reward = self.env.step(action, current_price)

			# NOTE: need to set next_state to None for 'done'
			next_state = self.get_state(t+1)

			# Experience Replay
			self.agent.replay_memory.append([current_state, action, reward, next_state, done])
			self.agent.experience_replay()

			current_state = next_state

			# Decay Exploration
			if self.epsilon > self.min_epsilon:		
				self.epsilon *= self.epsilon_decay
				self.epsilon = max(self.min_epsilon, self.epsilon)

			# Print progress and update stats of model
			self.print_progress(episode, t-start, done)
			self.update_stats(current_price, t, episode, reward)


	def train(self):
		# Training phase of DQN. 
		# Calculates the initial and terminal timesteps for each episodes
		# then runs the learning stage for each episode.

		for episode in range(self.train_episodes):
			episode_start = episode*self.episode_size
			episode_end = episode_start + self.episode_size - self.window_size - 1

			self.learn(episode, episode_start, episode_end)


	def validate(self):
		# Validation Phase of DQN. 
		# Evaluating performance on the set validation dataset (April).

		self.learn(-1, self.val_start, self.val_end)


	def update_stats(self, current_price, t, episode, reward):
		# Updates model stats dataframe with each timestep.
		#
		# param 	current_price 	Current Price
		# param 	t 				Current Timestep
		# param 	episode 		Episode Number 
		# 								-1 if validation phase
		# param 	reward 			Reward

		self.model_stats = self.model_stats.append({'price':current_price, 'ts':t, 
			'episode':episode, 'action':self.env.action_history[-1], 
			'inv':len(self.env.inventory), 'reward':reward, 'bank':self.env.bank, 
			'portfolio':self.env.portfolio}, ignore_index=True)


	def print_progress(self, episode, ts, done):
		# Prints current position and progress of the computation.
		#
		# param 	episode 		Episode Number 
		# param 	ts 				Current Timestep
		# param 	done 			Terminal timestep of episode

		if episode==-1:
			print ('Validation Phase: ')
			print ('Timestep: '+str(ts)+'/'+str(self.val_end-self.val_start))
			print ('')
			if done:
				print ('============================')
				print ('VALIDATION FINISHED')
				print ('Portfolio: '+str(self.env.portfolio))
				print ('Inventory: '+str(self.env.inventory))
				print ('============================')
				print ('')

		else:
			print ('Training Phase: ')
			print ('Episode: '+str(episode)+'/'+str(self.train_episodes))
			print ('Timestep: '+str(ts)+'/'+str(self.episode_size))
			print ('')

			# Print stats at end of episode
			if done:
				print ('============================')
				print ('EPISODE: '+str(episode))
				print ('Portfolio: '+str(self.env.portfolio))
				print ('Inventory: '+str(self.env.inventory))
				print ('============================')
				print ('')


	def save_results(self, fname):
		# Saves results in a pickle file.
		#
		# param 	fname 	Name of file

		with open('saved_data3/'+fname+'.pickle', 'wb') as handle:
			pickle.dump(self.model_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


	def plot_results(self):
		# Plot of actions and the change in portfolio in validation phase or at the
		# end of computation. 

		portfolio_history = self.env.portfolio_history
		plt.plot(range(len(portfolio_history)), portfolio_history)
		plt.title('validation portfolio')
		plt.show()

		action_history = self.env.action_history
		plt.plot(range(len(action_history)), action_history)
		plt.title('validation actions')
		plt.show()


	def run(self):
		# Runs entire program.

		self.set_variables()

		self.train()
		self.validate()

		self.save_results('force_explore')
		self.plot_results()


dqn = DQN()
dqn.run()




























