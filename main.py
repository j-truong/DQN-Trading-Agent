import numpy as np
import pandas as pd

from dqn_agent import DQN_agent
from env import Env 
from functions import get_model, plot_results, save_results

# NOTES
# time taken
# rewrite act, replay
# reset replay memory

data = pd.read_excel('data/eur-gbp.xlsx', names=['date','open','high','low','close','volume'])
data = data['close'][:10_000].to_numpy()
data = data.reshape(1, len(data))

# PARAMETERS
window_size = 10		# will be adjusted
episode_size = 60 	# two hours
#episodes = int( len(data)/(episode_size) )	
episodes = 2
batch_size = 32
initial_bank = 200
max_ts = 200

# Exploitation / Exporation parameters
epsilon = 1
epsilon_decay = 0.995
min_epsilon = 0.05

model_stats = pd.DataFrame(columns={'ts','episode','action','inv','profit', 'bank'})

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

		# Determie and execute action
		action = agent.act(current_state, epsilon)
		reward = env.step(action, current_state[0][0][-1], done)

		# Next timestep
		next_prices = data[:, t+1:t+1+window_size ]
		env_state = [env.bank, len(env.inventory)]
		next_state = [next_prices, env_state]

		# Experience Replay
		agent.replay_memory.append([current_state, action, reward, next_state, done])
		agent.experience_replay()

		# Update current state
		current_state = next_state

		print ('Episode: '+str(episode)+'/'+str(episodes))
		print (str(t)+' / '+str(episode_size))
		print (env.inventory)


		# Update model stats
		model_stats = model_stats.append({'ts':t, 'episode':episode, 'action':action, 
			'inv':len(env.inventory), 'profit':reward}, ignore_index=True)

		# Decay Exploration
		if epsilon > min_epsilon:		
			epsilon *= epsilon_decay
			epsilon = max(min_epsilon, epsilon)

		# Print stats at end of episode
		if done:
			print ('============================')
			print ('Episode: '+str(epsiode)+' / '+str(episodes))
			print ('Profit: '+str(env.profit))
			print ('Inventory: '+str(env.inventory))
			print ('============================')
			print ('')


plot_results(env)
save_results('test', model_stats)



































