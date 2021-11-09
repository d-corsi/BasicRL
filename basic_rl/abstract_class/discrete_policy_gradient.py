from basic_rl.abstract_class.basic_reinforcement import BasicReinforcement
import tensorflow as tf
import numpy as np
import abc

class DiscretePolicyGradient( BasicReinforcement, metaclass=abc.ABCMeta ):

	def __init__(self, env):
		self.action_space = env.action_space.n

		super().__init__(env)
		

	def plot_results( self, episode, ep_reward, ep_reward_mean, reward_list ):
		if self.verbose > 0: 
			print(f"{self.name} Episode: {episode:5.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}") 
		if self.verbose > 1: 
			np.savetxt(f"data/reward_{self.name}_{self.run_id}.txt", reward_list)
	

	def get_action(self, state):
		softmax_out = self.actor_network(state.reshape((1, -1)))
		selected_action = np.random.choice(self.action_space, p=softmax_out.numpy()[0])
		return selected_action, softmax_out[0][selected_action]


	def actor_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		reward = np.vstack(memory_buffer[:, 3])

		probability = self.actor_network(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)]
		probability = tf.expand_dims(tf.gather_nd(probability, action_idx), axis=-1)
		partial_objective = tf.math.log(probability) * ( reward - np.mean(reward) )

		return -tf.math.reduce_mean(partial_objective)

		
	def get_actor_model( self ):
		inputs = tf.keras.layers.Input(shape=self.input_shape)
		hidden_0 = tf.keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = tf.keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = tf.keras.layers.Dense(self.action_space, activation='softmax')(hidden_1)

		return tf.keras.Model(inputs, outputs)



