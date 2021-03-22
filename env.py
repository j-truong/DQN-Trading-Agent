
class Env:
	def __init__(self, initial_bank, pen):
		self.episode_rewards = []
		self.initial_bank = initial_bank
		self.neg_reward = 0
		self.penalty = pen

	def reset(self):
		# Reset all history of the environemnt at the start of each episode.

		self.action_history = []
		self.inventory = []
		self.bank = self.initial_bank
		self.bank_history = []

	def executable(self,action, current_price):
		# Determines whether an action is executable given the action and 
		# corresponding amount. This is checked against the environment's bank
		# and inventory.
		#
		# param	 action 		Action
		# param  current_price 	Current Price of data
		# output executable 	Boolean of whether the action is executable:
		# 						True = executable, False = not executable

		if action < 0: # BUY
			if abs(action*current_price) > self.bank:
				return False

		if action > 0: # SELL
			if action > len(self.inventory):
				return False

		return True

	def step(self,action,current_price,done):
		# Exceute determined action in environment and updates history.
		# BUY = <0, HOLD = 0, SELL = >0
		#
		# Comments: 	- can change inventory selection from queue to min/max
		#				- reward can be actual change and not only positive
		# 
		# param  action 		Action taken
		# param  current_price 	Current price
		# param  done 			Boolean if final timestep (ts) in episode:
		# 						True = is final ts, False: is not final ts
		# output reward 		Reward 
		
		if not self.executable(action, current_price):
			self.action_history.append(0)
			#return self.neg_reward
			return -self.penalty

		else:
			self.bank += action*current_price
			self.action_history.append(action)
			self.bank_history.append(self.bank)

			reward = 0
			if action < 0: # BUY
				for _ in range(abs(action)):
					self.inventory.append(current_price)

			elif action > 0: # SELL
				for _ in range(action):
					prev_price = self.inventory.pop(0)
					reward += current_price - prev_price

			if done:   # Append final episodic profit
				self.episode_rewards.append(self.bank)

			# only return reward if positive (adjustable)
			return (reward if reward > 0 else 0)

















