import numpy as np
import pandas as pd
import math

from dqn_agent import DQN_agent
from env import Env 
from functions import get_model, plot_results, save_results

# NOTES
# time taken
# rewrite act, replay
# reset replay memory

data = pd.read_excel('data/eur-gbp.xlsx', names=['date','open','high','low','close','volume'])
data = data['close'].to_numpy()
data = data.reshape(1, len(data))

# PARAMETERS
window_size = 10		# will be adjusted
episode_size = 120 	# two hours
episodes = math.floor( data.shape[1]/(episode_size) )	
episodes = 1
batch_size = 32
initial_bank = 200
max_ts = 200

# Exploitation / Exporation parameters
epsilon = 1
epsilon_decay = 0.99
min_epsilon = 0.1

model_stats = pd.DataFrame(columns={'price','ts','episode','action','inv','profit', 'bank'})

model = get_model(window_size)

agent = DQN_agent(model, batch_size, max_ts)
env = Env(initial_bank)

for episode in range(episodes):

	# reset environment
	env.reset()		
	# Current timestep
	episode_start = episode*episode_size
	current_prices = data[:, episode_start:episode_start + window_size]
	env_state = [env.bank, len(env.inventory)]
	current_state = [current_prices, env_state]


	done = False
	for t in range(episode_start, episode_start + episode_size - window_size - 1):

		current_price = current_state[0][0][-1]

		# Determie and execute action
		action = agent.act(current_state, epsilon)
		execute = env.executable(action, current_price)
		if not execute:
			action = agent.explore(current_state)
		reward = env.step(action, current_price, done)

		# Next timestep
		next_prices = data[:, t+1:t+1+window_size ]
		env_state = [env.bank, len(env.inventory)]
		next_state = [next_prices, env_state]

		# Experience Replay
		agent.replay_memory.append([current_state, action, reward, next_state, done])
		agent.experience_replay()

		# Update current state and  model stats
		current_state = next_state
		model_stats = model_stats.append({'price':current_price, 'ts':t, 'episode':episode, 
			'action':action, 'inv':len(env.inventory), 'profit':reward, 'bank':env.bank}, ignore_index=True)

		# Decay Exploration
		if epsilon > min_epsilon:		
			epsilon *= epsilon_decay
			epsilon = max(min_epsilon, epsilon)

		# Print progress
		print ('Episode: '+str(episode)+'/'+str(episodes))
		print (str(t-episode_start)+' / '+str(episode_size))

		# Print stats at end of episode
		if done:
			print ('============================')
			print ('Episode: '+str(epsiode)+' / '+str(episodes))
			print ('Profit: '+str(env.profit))
			print ('Inventory: '+str(env.inventory))
			print ('============================')
			print ('')


plot_results(env)
save_results('forced_explore2', model_stats)



































