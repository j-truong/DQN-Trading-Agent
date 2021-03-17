

class Env:
	def __init__(self, initial_bank):
		self.episode_rewards = []
		self.initial_bank = initial_bank
		self.neg_reward = 0

	def reset(self):
		# Reset all history of the environemnt at the start of each episode.

		self.action_history = []
		self.inventory = []
		self.bank = self.initial_bank
		self.profit_history = []

	def executable(self,action, current_price):

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
		# param action 			Action taken
		# param current_price 	Current price
		
		if not self.executable(action, current_price):
			self.action_history.append(action)
			return self.neg_reward

		else:
			self.action_history.append(action)
			reward = 0

			if action < 0: # BUY
				self.bank += action*current_price
				for _ in range(abs(action)):
					self.inventory.append(current_price)
				print ('buy')

			elif action > 0: # SELL
				for _ in range(action):
					prev_price = self.inventory.pop(0)	# as a queue
					reward += current_price - prev_price
				self.bank += reward

			self.profit_history.append(reward)

			# Append final episodic profit
			if done:
				self.episode_rewards.append(self.bank)

			# only return reward if positive
			return (reward if reward > 0 else 0)

















