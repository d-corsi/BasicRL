from basic_rl.abstract_class.continuous_policy_gradient import ContinuousPolicyGradient
from basic_rl.abstract_class.discrete_policy_gradient import DiscretePolicyGradient
import numpy as np
import tensorflow as tf

class DiscreteActorCritic( DiscretePolicyGradient ):

	train_critic = True

	def __init__(self, env, **kwargs):
		super().__init__(env)

		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)


	def actor_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		reward = np.vstack(memory_buffer[:, 3])

		probability = self.actor_network(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)]
		probability = tf.expand_dims(tf.gather_nd(probability, action_idx), axis=-1)
		partial_objective = tf.math.log(probability) * ( reward - self.critic_network(state) )

		return -tf.math.reduce_mean(partial_objective)


class ContinuousActorCritic( ContinuousPolicyGradient ):

	train_critic = True

	def __init__(self, env, **kwargs):
		super().__init__(env)

		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)


	def actor_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = np.vstack(memory_buffer[:, 1])
		reward = np.vstack(memory_buffer[:, 3])

		mu = self.actor_network(state)
		pdf_value = tf.sqrt(1/(2 * np.pi * self.sigma**2)) * tf.exp(-(action - mu)**2/(2 * self.sigma**2))
		pdf_value = tf.math.reduce_mean(pdf_value, axis=1, keepdims=True)
		partial_objective = tf.math.log(pdf_value) * ( reward - self.critic_network(state) )

		return -tf.math.reduce_mean(partial_objective)
