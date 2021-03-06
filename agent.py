import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, model, batch_size, max_ts):
		self.action_size = 3
		self.gamma = 0.95
		self.batch_size = batch_size
		self.max_ts = max_ts
		self.replay_memory = deque(maxlen=1_000)
		self.min_replay_memory = 100

		self.model = model
		self.target_model = model

		self.target_counter = 0

		self.buy_limit = 0.5 		# Bank Buy Limit
		self.sell_limit = 1			# Inventory Sell Limit


	def explore(self, state):
		# Force exploration in deciding an action with upper and lower bounds in how
		# much the agent can buy or sell. Both limits fixed in initialisation. 
		#
		# param  state 		Current state
		# output action 	Action

		bank = state[1][0]
		inv = state[1][1]
		action = np.random.randint( -self.buy_limit*bank, self.sell_limit*inv)

		return action


	def act(self,state,epsilon):
		# Decide action based on epsilon; the parameter for exploitation
		# and exploration. 
		#
		# param 	current_state 	Current State of environment
		# param 	epsilon			Epsilon
		# output 	action 			Action Taken

		# Exploration
		if np.random.random() < epsilon:
			action = self.explore(state)

		# Exploitation
		else: 
			pred = self.model.predict({"price_input":np.array(state[0]), 
				"env_input":np.array([state[1]])})
			action = int(pred)

		return action


	def experience_replay(self):
		# Trains NN in minibatches where the inputs is a randomly selected state in the
		# replay memory and the outputs are the q values for each action that can be taken
		# for that state.
		#
		# https://www.youtube.com/watch?v=xVkPh9E9GfE&ab_channel=deeplizard
		#
		# param max_ts 	Max target steps

		if len(self.replay_memory) < self.min_replay_memory:
			return

		minibatch = random.sample(self.replay_memory, self.batch_size)
		self.target_counter += self.batch_size

		price_input = []
		env_input = []
		q_values = []
		for (current_state, action, reward, next_state, done) in minibatch:

			# Calculate target Q value
			if done:
				target_q = reward		# because no new_state (final instance in episode)
			else:
				# Bellman's Equation
				next_qs = self.target_model.predict({"price_input":np.array(next_state[0]), 
					"env_input":np.array([next_state[1]])})
				target_q  = reward + self.gamma*np.max(next_qs)

			price_input.append(current_state[0][0].tolist())
			env_input.append(current_state[1])
			q_values.append([target_q])

		price_input = np.array(price_input)
		env_input = np.array(env_input)
		q_values = np.array(q_values)
		

		history = self.model.fit({"price_input":price_input, "env_input":env_input},
			{"action_output":q_values}, verbose=0)

		if self.target_counter > self.max_ts:
			self.target_model.set_weights( self.model.get_weights() )
			self.target_counter = 0





























