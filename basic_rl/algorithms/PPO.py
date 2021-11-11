from basic_rl.abstract_class.continuous_policy_gradient import ContinuousPolicyGradient
from basic_rl.abstract_class.discrete_policy_gradient import DiscretePolicyGradient
import numpy as np
import tensorflow as tf

class DiscretePPO( DiscretePolicyGradient ):

	train_critic = True
	temporal_difference_method = True
	use_batch = True

	def __init__(self, env, **kwargs):
		super().__init__(env)

		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)


	def actor_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		action_prob = np.vstack(memory_buffer[:, 2])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		baseline = self.critic_network(state)
		adv = self.temporal_difference(reward, new_state, done) - baseline # Advantage = TD - baseline

		prob = self.actor_network(state)

		action_idx = [[counter, val] for counter, val in enumerate(action)] #Trick to obatin the coordinates of each desired action
		prob = tf.expand_dims(tf.gather_nd(prob, action_idx), axis=-1)
		r_theta = tf.math.divide(prob, action_prob) #prob/old_prob

		clip_val = 0.2
		obj_1 = r_theta * adv
		obj_2 = tf.clip_by_value(r_theta, 1-clip_val, 1+clip_val) * adv
		partial_objective = tf.math.minimum(obj_1, obj_2)

		return -tf.math.reduce_mean(partial_objective)


class ContinuousPPO( ContinuousPolicyGradient ):

	train_critic = True
	temporal_difference_method = True
	use_batch = True

	def __init__(self, env, **kwargs):
		super().__init__(env)

		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

	
	def actor_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = np.vstack(memory_buffer[:, 1])
		mu = np.vstack(memory_buffer[:, 2])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		baseline = self.critic_network(state)
		adv = self.temporal_difference(reward, new_state, done) - baseline # Advantage = TD - baseline

		predictions_mu = self.actor_network(state)

		prob = tf.sqrt(1/(2 * np.pi * self.sigma**2)) * tf.exp(-(action - predictions_mu)**2/(2 * self.sigma**2))
		old_prob = tf.sqrt(1/(2 * np.pi * self.sigma**2)) * np.math.e ** (-(action - mu)**2/(2 * self.sigma**2))
		prob = tf.math.reduce_mean(prob, axis=1, keepdims=True)
		old_prob = tf.math.reduce_mean(old_prob, axis=1, keepdims=True)

		r_theta = tf.math.divide(prob, old_prob.numpy()) #prob/old_prob

		clip_val = 0.2
		obj_1 = r_theta * adv
		obj_2 = tf.clip_by_value(r_theta, 1-clip_val, 1+clip_val) * adv
		partial_objective = tf.math.minimum(obj_1, obj_2)

		return -tf.math.reduce_mean(partial_objective)
