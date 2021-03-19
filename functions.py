import pickle
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, concatenate, Dense, Flatten

def get_model(window):

	# Inputs
	price_input = Input(shape=(window,), name='price_input')
	env_input = Input(shape=(2,), name='env_input')

	# Adjsusted model
	price_layer1 = Dense(32, activation='relu', name='price_layer1')(price_input)
	price_layer2 = Dense(16, activation='relu', name='price_layer2')(price_layer1)
	price_final = Flatten(name='price_flatten')(price_layer2)

	# Fixed layers and output
	concat_layer = concatenate([price_final, env_input], name='concat_layer')
	fixed_layer1 = Dense(8, activation='relu', name='fixed_layer1')(concat_layer)
	fixed_layer2 = Dense(4, activation='relu', name='fixed_layer2')(fixed_layer1)

	action_output = Dense(1, activation='linear', name='action_output')(fixed_layer2)

	model = Model(inputs=[price_input, env_input],
	              outputs=[action_output])

	model.compile(optimizer='SGD', loss={'action_output':'mse'})

	return model

def save_results(fname, data):
	# save data

	with open('saved_data/'+fname+'.pickle', 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_results(env):
	profit_history = env.profit_history
	plt.plot(range(len(profit_history)), profit_history)
	plt.title('profit')
	plt.show()

	profit_history = np.cumsum(env.profit_history)
	plt.plot(range(len(profit_history)), profit_history)
	plt.title('acc profit')
	plt.show()

	action_history = env.action_history
	plt.plot(range(len(action_history)), action_history)
	plt.title('actions')
	plt.show()









