
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import time

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, concatenate, Dense, Flatten
from tensorflow.keras.layers import Conv1D, AveragePooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model

from agent import Agent
from env import Env 


class DQN:
	def __init__(self):
		# Initialise model parameters. Episode size, intial bank will be fixed. 
		# Training data chosen to be Jan, Feb, Mar.
		# Validation data chosen to be Apr. 

		# RL parameters
		self.window_size = 5			# chosen
		self.episode_size = 120 		
		self.train_episodes = 386
		self.batch_size = 32
		self.initial_bank = 200
		self.max_ts = 200				# chosen

		# Exploitation / Exporation parameters
		self.epsilon = 1
		self.epsilon_decay = 0.99999	# chosen
		self.min_epsilon = 0.1

		# Fixed validation and test timesteps
		self.val_start = 46340
		self.val_end = 60680
		self.test_end = 121589

		# Print Progress
		self.print_name = None
		self.print_value = None

		# Time
		self.start_time = None


	def fixed_model(self):
		# A pre-determined and fixed model for experimentation in determining
		# Reinforcement Learning parameters.
		#
		# output 	model 	Fixed model

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

			# Next state
			if not done:
				next_state = self.get_state(t+1)
			else:
				next_state = None

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
		# 							- Validation Phase: -1
		# 							- Test Phase: 		-2	
		# param 	reward 			Reward

		time_taken = self.start_time - time.time()
		#hl, node1, node2, opt = self.print_value

		self.model_stats = self.model_stats.append({'price':current_price, 'ts':t, 
			'episode':episode, 'action':self.env.action_history[-1], 'inv':len(self.env.inventory), 
			'reward':reward, 'bank':self.env.bank, 'portfolio':self.env.portfolio, 'time':time_taken},

			#'hidden_layer':hl, 'layer_size1':node1, 'layer_size2':node2, 'opt':opt}, 
			ignore_index=True)


	def print_progress(self, episode, ts, done):
		# Prints current position and progress of the computation.
		#
		# param 	episode 		Episode Number 
		# param 	ts 				Current Timestep
		# param 	done 			Terminal timestep of episode

		# VALIDATION PHASE
		if episode==-1:
			print ('Validation Phase: ')
			print ('Timestep: '+str(ts)+'/'+str(self.val_end-self.val_start))

			# print parameters
			if self.print_name and self.print_value:
				for name, value in zip(self.print_name, self.print_value):
					print (name+': '+str(value))
			print ('')

			if done:
				print ('============================')
				print ('VALIDATION FINISHED')
				print ('Portfolio: '+str(self.env.portfolio))
				print ('Inventory: '+str(self.env.inventory))
				print ('============================')
				print ('')

		# TEST PHASE
		elif episode==-2:
			print ('Test Phase: ')
			print ('Timestep: '+str(ts)+'/'+str(self.test_end-self.val_end))
			print ('')

			if done:
				print ('============================')
				print ('TEST FINISHED')
				print ('Portfolio: '+str(self.env.portfolio))
				print ('Inventory: '+str(self.env.inventory))
				print ('============================')
				print ('')
				

		# TRAINING PHASE
		else:
			print ('Training Phase: ')
			print ('Episode: '+str(episode)+'/'+str(self.train_episodes))
			print ('Timestep: '+str(ts)+'/'+str(self.episode_size))

			# print parameters
			if self.print_name and self.print_value:
				for name, value in zip(self.print_name, self.print_value):
					print (name+': '+str(value))
			print ('')

			# Print stats at end of episode
			if done:
				print ('============================')
				print ('EPISODE: '+str(episode))
				print ('Portfolio: '+str(self.env.portfolio))
				print ('Inventory: '+str(self.env.inventory))
				print ('============================')
				print ('')


	def save_results(self,folder,fname):
		# Saves results in a pickle file.
		#
		# param 	fname 	Name of file

		with open(folder+'stats/'+fname+'.pickle', 'wb') as handle:
			pickle.dump(self.model_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open(folder+'replay_memory/'+fname+'.pickle', 'wb') as handle:
			pickle.dump(self.agent.replay_memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

		model = self.agent.model
		model.save(folder+'model/'+fname)
		

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


	def get_ANN(self,hidden_layer,nodes1,nodes2,opt):
		# Creates a small and simple Artificial Neural Network (ANN) with the given parameters. 
		# Consists of only hidden layers - dropout can also be considered.
		#
		# param 	hidden_layer 	Amount of hidden layers in ANN
		# param 	nodes1 			Layer size of first hidden layer
		# param 	nodes2 			Layer size of second hidden layer (if feasible)
		# param 	opt 			Optimizer of ANN
		# output 	model 			ANN model

		# Input Layers
		price_input = Input(shape=(self.window_size,), name='price_input')
		env_input = Input(shape=(2,), name='env_input')

		# Adjsustable Hidden Layers
		price_layer = Dense(nodes1, activation='relu', name='price_layer1')(price_input)
		for _ in range(hidden_layer-1):
			price_layer = Dense(nodes2, activation='relu', name='price_layer2')(price_layer)

		# Dropout
		#if dropout:
		#	price_final = Dropout(0.5, name='dropout')(price_final)

		price_final = Flatten(name='price_flatten')(price_layer)

		# Fixed layers
		concat_layer = concatenate([price_final, env_input], name='concat_layer')
		fixed_layer1 = Dense(8, activation='relu', name='fixed_layer1')(concat_layer)
		fixed_layer2 = Dense(4, activation='relu', name='fixed_layer2')(fixed_layer1)

		# Output Layer
		action_output = Dense(1, activation='linear', name='action_output')(fixed_layer2)

		model = Model(inputs=[price_input, env_input], outputs=[action_output])
		model.compile(optimizer=opt, loss={'action_output':'mse'})

		return model


	def get_CNN(self,layers,filters1,filters2,opt):
		# Creates a small and simple Convolutional Neural Network (CNN) with the given parameters. 
		# Consists of Convolutional layers, average pooling - dropout can be considered.
		#
		# param 	hidden_layer 	Amount of hidden layers in ANN
		# param 	filters1 		Filter size of first hidden layer
		# param 	filters2 		Filter size of second hidden layer (if feasible)
		# param 	opt 			Optimizer 
		# output 	model 			CNN model

		# Input Layers
		price_input = Input(shape=(self.window_size,1,), name='price_input')
		env_input = Input(shape=(2,), name='env_input')

		# Adjsustable Hidden Layers
		price_layer = Conv1D(filters=filters1, kernel_size=2, activation='relu', name='price_layer1')(price_input)
		for _ in range(layers-1):
			price_layer = Conv1D(filters=filters2, kernel_size=2, activation='relu', name='price_layer2')(price_layer)
		
		# Average Pooling Layers
		price_layer = AveragePooling1D(pool_size=2)(price_layer)

		# Dropout
		#if dropout:
		#	price_final = Dropout(0.1, name='dropout')(price_final)

		price_final = Flatten(name='price_flatten')(price_layer)

		# Fixed layers
		concat_layer = concatenate([price_final, env_input], name='concat_layer')
		fixed_layer1 = Dense(8, activation='relu', name='fixed_layer1')(concat_layer)
		fixed_layer2 = Dense(4, activation='relu', name='fixed_layer2')(fixed_layer1)

		# Output Layer
		action_output = Dense(1, activation='linear', name='action_output')(fixed_layer2)

		model = Model(inputs=[price_input, env_input], outputs=[action_output])
		model.compile(optimizer=opt, loss={'action_output':'mse'})

		return model


	def get_RNN(self,layers,units1,units2,opt):
		# Creates a small and simple Convolutional Neural Network (CNN) with the given parameters. 
		# Consists of Convolutional layers, average pooling, and dropout can be considered.
		#
		# param 	hidden_layer 	Amount of hidden layers in ANN
		# param 	nodes1 			Layer size of first hidden layer
		# param 	nodes2 			Layer size of second hidden layer (if feasible)
		# param 	opt 			Optimizer of ANN
		# output 	model 			ANN model

		# Input Layers
		price_input = Input(shape=(self.window_size,1,), name='price_input')
		env_input = Input(shape=(2,), name='env_input')

		# Adjsustable Hidden Layers
		if layers == 1:
			price_layer = LSTM(units1, activation='relu', name='price_layer1')(price_input)
		elif layers > 1:
			price_layer = LSTM(units1, return_sequences=True, activation='relu', name='price_layer1')(price_input)
			for _ in range(layers-1):
				price_layer = LSTM(units2, activation='relu', name='price_layer2')(price_layer)
		price_final = Flatten(name='price_flatten')(price_layer)

		# Dropout
		#if dropout:
		#	price_final = Dropout(name='dropout')(price_final)

		# Fixed layers
		concat_layer = concatenate([price_final, env_input], name='concat_layer')
		fixed_layer1 = Dense(8, activation='relu', name='fixed_layer1')(concat_layer)
		fixed_layer2 = Dense(4, activation='relu', name='fixed_layer2')(fixed_layer1)

		# Output Layer
		action_output = Dense(1, activation='linear', name='action_output')(fixed_layer2)

		model = Model(inputs=[price_input, env_input], outputs=[action_output])
		model.compile(optimizer=opt, loss={'action_output':'mse'})

		return model


	def run_model(self,model_type,stat_columns,hidden_layer,layer_size1,layer_size2,opt):
		# Run the training and validation phase of the model with the selected combination.
		# Model data is also saved.
		#
		# param 	model_type		Neural Network type to be used in DQN model
		# 							'ANN' = Artificial Neural Network
		#							'CNN' = Convolutional Neural Network
		#							'RNN' = Recurrent Neural Network
		# param 	stat_columns	Recorded statistics of the model 
		# param 	hidden_layer 	Amount of hidden layers 
		# param 	layer_size1 	Node size of first hidden layer
		# param 	layer_size2 	Node size of second hidden layer
		# param 	opt 			Optimizer

		self.start_time = time.time()
		self.model_stats = pd.DataFrame(columns=stat_columns)

		# Retrieve model combination dependant on model type
		if model_type == 'ANN':
			model = self.get_ANN(hidden_layer, layer_size1, layer_size2, opt)
			folder = 'saved_data3/DQN_ANN/train/'

		elif model_type == 'CNN':
			model = self.get_CNN(hidden_layer, layer_size1, layer_size2, opt)
			folder = 'saved_data3/DQN_CNN/train/'

		elif model_type == 'RNN':
			model = self.get_RNN(hidden_layer, layer_size1, layer_size2, opt)
			folder = 'saved_data3/DQN_RNN/train/'

		else:
			raise NameError('Unfamilar model type')
		
		# Initialise agent and environment
		self.agent = Agent(model, self.batch_size, self.max_ts)
		self.env = Env(self.initial_bank)

		# Training and validation phase
		self.train()
		self.validate()

		# Save results
		fname = str(hidden_layer)+'_'+str(layer_size1)+'_'+str(layer_size2)+'_'+str(opt)
		self.save_results(folder, fname)


	def run(self, model_type):
		# Runs entire program.
		#
		# param 	model_type 		Neural Network type to be used in DQN model
		# 							'ANN' = Artificial Neural Network
		#							'CNN' = Convolutional Neural Network
		#							'RNN' = Recurrent Neural Network

		# Read data
		df = pd.read_excel('data/eur-gbp.xlsx', 
			names=['date','open','high','low','close','volume'])
		data = df['close'].to_numpy()
		self.data = data.reshape(1, len(data))	

		# Record model progress
		self.print_name = ['hidden_layer','layer_size1','layer_size2','opt']
		stat_columns = {'price','ts','episode','action','inv','reward', 'bank', 'portfolio','time',
			'hidden_layer','layer_size1','layer_size2','opt'}

		# Adjusted parameters
		hidden_layers = [2]
		layer_sizes = [8,32]
		optimizers = ['SGD','Adam']

		# Combinations of DQN model
		for hidden_layer in hidden_layers:
			for layer_size1 in layer_sizes:
				for opt in optimizers:

					if hidden_layer == 1:
						layer_size2 = 0
						self.print_value = [hidden_layer, layer_size1, layer_size2, opt]
						self.run_model(model_type, stat_columns, hidden_layer, layer_size1, layer_size2, opt)

					elif hidden_layer == 2:
						for layer_size2 in layer_sizes:
							#if layer_size1 >= layer_size2:
							if (layer_size1==8) and (layer_size2==8):
								self.print_value = [hidden_layer, layer_size1, layer_size2, opt]
								self.run_model(model_type, stat_columns, hidden_layer, layer_size1, layer_size2, opt)
								#pass 
							else:
								#self.print_value = [hidden_layer, layer_size1, layer_size2, opt]
								#self.run_model(model_type, stat_columns, hidden_layer, layer_size1, layer_size2, opt)
								pass
								

	def test(self, model_folder, fname):
		# Test trained and validated model on test dataset.
		#
		# param 	model_folder 	Name of folder the model resides
		# param 	fname 			File name

		# Read Data
		df = pd.read_excel('data/eur-gbp.xlsx', 
			names=['date','open','high','low','close','volume'])
		data = df['close'].to_numpy()
		self.data = data.reshape(1, len(data))	

		# Load trained model and replay memory
		trained_model = load_model('saved_data3/'+model_folder+'/train/model/'+fname)
		memory_name = 'saved_data3/'+model_folder+'/train/replay_memory/'+fname+'.pickle'
		with open(memory_name, 'rb') as handle:
		    memory = pickle.load(handle)

		# Record statistics and model progress
		self.start_time = time.time()
		stat_columns = {'price','ts','episode','action','inv','reward','bank','portfolio','time'}
		self.model_stats = pd.DataFrame(columns=stat_columns)

		# Initialise agent and environment
		self.env = Env(self.initial_bank)
		self.agent = Agent(trained_model, self.batch_size, self.max_ts)
		self.agent.replay_memory = memory

		# Continue learning in testing phase
		self.learn(-2, self.val_end, self.test_end)

		# Save data
		folder = 'saved_data3/'+model_folder+'/test/'
		self.save_results(folder, fname)


	def retrain_model(self, model, folder, fname):
		# Train, validate and test a model to a given dataset. 

		# Record statistics and model progress
		self.start_time = time.time()
		stat_columns = {'price','ts','episode','action','inv','reward','bank','portfolio','time'}
		self.model_stats = pd.DataFrame(columns=stat_columns)

		self.agent = Agent(model, self.batch_size, self.max_ts)
		self.env = Env(self.initial_bank)

		# Training and validation phase
		self.train()
		self.validate()
		self.learn(-2, self.val_end, self.test_end)

		# Save results
		with open('saved_data/'+folder+'/'+fname+'.pickle', 'wb') as handle:
			pickle.dump(self.model_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


	def retrain(self, data_name):
		# After selecting parameters for each DQN variation, this method will retrain the same 
		# parameters onto a different dataset. 
		# param 	data_name 	Name of Dataset

		df = pd.read_excel('data/'+data_name+'.xlsx', 
			names=['date','open','high','low','close','volume'])
		data = df['close'].to_numpy()
		self.data = data.reshape(1, len(data))	

		self.print_name  = ['model type']

		# DQN_ANN 
		self.print_value  = ['DQN_ANN']
		ANN = self.get_ANN(2,32,32,'Adam')
		self.retrain_model(ANN, data_name, 'DQN_ANN 2_32_32_Adam')

		# DQN_CNN 
		self.print_value  = ['DQN_CNN']
		CNN = self.get_CNN(2,32,32,'Adam')
		self.retrain_model(CNN, data_name, 'DQN_CNN 2_32_32_Adam')
		self.print_value  = ['DQN_CNN']

		# DQN_RNN 
		self.print_value  = ['DQN_RNN']
		RNN = self.get_RNN(2,32,8,'SGD')
		self.retrain_model(RNN, data_name, 'DQN_RNN 2_32_8_SGD')



dqn = DQN()
#dqn.run('RNN')
#dqn.test('DQN_RNN','2_32_8_SGD')
dqn.retrain('eur-chf')





























