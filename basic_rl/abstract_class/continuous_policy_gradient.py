from basic_rl.abstract_class.basic_reinforcement import BasicReinforcement
import tensorflow as tf
import numpy as np
import abc

class ContinuousPolicyGradient( BasicReinforcement, metaclass=abc.ABCMeta ):

	def __init__(self, env):
		self.action_space = env.action_space.shape[0]
		self.output_range = [env.action_space.low, env.action_space.high]

		super().__init__(env)
		

	def plot_results( self, episode, ep_reward, ep_reward_mean, reward_list ):
		if self.verbose > 0: 
			print(f"{self.name} Episode: {episode:5.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, sigma: {self.sigma:0.2f}")
		if self.verbose > 1: 
			np.savetxt(f"data/reward_{self.name}_{self.run_id}.txt", reward_list)


	def get_action(self, state):
		mu = self.actor_network(state.reshape((1, -1)))
		action = np.random.normal(loc=mu, scale=self.sigma)
		return action[0], mu[0]


	def actor_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = np.vstack(memory_buffer[:, 1])
		reward = np.vstack(memory_buffer[:, 3])

		mu = self.actor_network(state)
		pdf_value = tf.sqrt(1/(2 * np.pi * self.sigma**2)) * tf.exp(-(action - mu)**2/(2 * self.sigma**2))
		pdf_value = tf.math.reduce_mean(pdf_value, axis=1, keepdims=True)
		partial_objective = tf.math.log(pdf_value) * ( reward - np.mean(reward) )

		return -tf.math.reduce_mean(partial_objective)


	def get_actor_model( self ):
		# Initialize weights between -3e-3 and 3-e3
		last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		inputs = tf.keras.layers.Input(shape=self.input_shape)
		hidden_0 = tf.keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = tf.keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = tf.keras.layers.Dense(self.action_space, activation='sigmoid', kernel_initializer=last_init)(hidden_1)

		# Fix output range with the range of the action
		outputs = outputs * (self.output_range[1] - self.output_range[0]) + self.output_range[0]

		return tf.keras.Model(inputs, outputs)
