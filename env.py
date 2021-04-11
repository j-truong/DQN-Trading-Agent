
class Env:
	def __init__(self, initial_bank, pen_reward=0, percent=0, fee=0):
		self.initial_bank = initial_bank

		# Negative Rewards
		self.pen_reward = 0				# Penalise action
		self.trans_percent = percent 	# Transaction percentage fee
		self.trans_fee = fee 			# Fixed Transaction fee

	def reset(self):
		# Reset all history of the environemnt at the start of each episode.

		self.action_history = []
		self.inventory = []
		self.bank = self.initial_bank
		self.portfolio = self.initial_bank
		self.portfolio_history = []

	def executable(self, action, current_price):
		# Determines whether an action is executable given the action and 
		# corresponding amount. This is checked against the environment's bank
		# and inventory.
		#
		# param	 	action 			Action
		# param  	current_price 	Current Price of data
		# output 	executable 		Boolean of whether the action is executable:
		# 								True = executable, False = not executable

		if action < 0: # BUY
			if abs(action*current_price) > self.bank:
				return False

		if action > 0: # SELL
			if action > len(self.inventory):
				return False

		return True


	def step(self, action, current_price):
		# Exceute determined action in environment and updates history.
		# BUY = <0, HOLD = 0, SELL = >0
		#
		# Comments: 	- can change inventory selection from queue to min/max
		#				- reward can be actual change and not only positive
		# 
		# param  	action 			Action taken
		# param  	current_price 	Current price
		# param  	done 			Boolean if final timestep (ts) in episode:
		# 								True = is final ts, False: is not final ts
		# output 	reward 			Reward 
		
		if not self.executable(action, current_price):
			self.action_history.append(0)
			return self.pen_reward

		else:
			# Cost of transaction
			if action != 0:
				transaction = action*current_price
				additional_cost = - abs(transaction*self.trans_percent) - self.trans_fee
			else:
				transaction = 0
				additional_cost = 0
			self.bank += transaction + additional_cost


			# Update Environment
			reward = additional_cost
			if action < 0: # BUY
				for _ in range(abs(action)):
					self.inventory.append(current_price)

			elif action > 0: # SELL
				for _ in range(action):
					prev_price = self.inventory.pop(0)
					reward += current_price - prev_price


			# Update stats 
			self.action_history.append(action)
			self.portfolio = self.bank + (current_price*len(self.inventory))
			self.portfolio_history.append(self.portfolio)

			# only positive reward
			#return (reward if reward > 0 else 0)
			return reward

















