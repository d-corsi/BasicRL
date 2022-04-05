from basic_rl.gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf


class MonteCarloPPO( ReinforcementLearning ):

	"""
	Class that inherits from ReinforcementLearning to implements the PPO algorithm with a Monte Carlo approach, 
	the original paper can be found here:
	https://arxiv.org/abs/1707.06347 [1]

	[1] Proximal Policy Optimization Algorithms, 
		Schulman et al., 
		arXiv preprint arXiv:1707.06347, 2017

	"""


	# Constructor of the class
	def __init__( self, env, verbose, str_mod="mcPPO", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )

		#
		self.memory_size = None
		self.gamma = 0.99
		self.trajectory_update = 10
		self.critic_epoch = 40
		self.critic_batch_size = 128
		self.layers = 2
		self.nodes = 32
		self.layers_critic = self.layers
		self.nodes_critic = self.nodes

		# 
		self.relevant_params = {
			'gamma' : 'gamma',
			'trajectory_update' : 'tu',
			'critic_epoch' : 'ce',
			'critic_batch_size' : 'cbs'
		}

		# Override the default parameters with kwargs
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		#
		self.memory_buffer = deque( maxlen=self.memory_size )

		#
		self.actor = self.generate_model(self.input_shape, self.action_space, \
			layers=self.layers, nodes=self.nodes, last_activation='softmax')
		self.critic = self.generate_model(self.input_shape, layers=self.layers_critic, nodes=self.nodes_critic)

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()
		self.critic_optimizer = tf.keras.optimizers.Adam()


	# Mandatory method to implement for the ReinforcementLearning class, decide the 
	# update frequency and some variable update for on/off policy algorithms
	# (i.e., eps_greedy, buffer, ...)
	def network_update_rule( self, episode, terminal ):

		# Update of the networks for PPO!
		# - Performed every <trajectory_update> episode
		# - clean up of the buffer at each update (on-policy).
		# - Only on terminal state (collect the full trajectory)
		if terminal and episode % self.trajectory_update == 0:
			self.update_networks(np.array(self.memory_buffer, dtype=object))
			self.memory_buffer.clear()


	# Application of the gradient with TensorFlow and based on the objective function
	def update_networks( self, memory_buffer ):

		# Find the end of each trajectory to compute the montecarlo returns
		done = np.vstack(memory_buffer[:, 5]); counter = 0

		# Replace the reward with the discoutned returs in the buffer, in the MC approach
		# this is a foundamental part to preserve the concept of single trajectory. The discount
		# is computed separately for each trajectory
		for i in np.where(done == True)[0]:
			memory_buffer[:, 3][counter:i+1] = self.discount_reward(memory_buffer[:, 3][counter:i+1])
			counter = i+1

		# Critic update (repeated epoch times on a batch, fixed):
		for _ in range( self.critic_epoch ):

			# Computing a random sample of elements from the batch for the training,
			# randomized at each epoch
			idx = np.random.randint(memory_buffer.shape[0], size=self.critic_batch_size)
			training_batch = memory_buffer[idx]

			#
			with tf.GradientTape() as critic_tape:

				# Compute the objective function, compute the gradient information and apply the
				# gradient with the optimizer
				critic_objective_function = self.critic_objective_function( training_batch )
				critic_gradient = critic_tape.gradient( critic_objective_function, self.critic.trainable_variables )
				self.critic_optimizer.apply_gradients( zip(critic_gradient, self.critic.trainable_variables) )

		# Actor update (repeated 1 time for each call):
		with tf.GradientTape() as actor_tape:

			# Compute the objective function, compute the gradient information and apply the
			# gradient with the optimizer
			actor_objective_function = self.actor_objective_function( memory_buffer )
			actor_gradient = actor_tape.gradient(actor_objective_function, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients( zip(actor_gradient, self.actor.trainable_variables) )


	# Compute the discounted reward, require the full trajectory
	def discount_reward(self, rewards):

		#
		sum_reward = 0
		discounted_rewards = []

		# Iterate on the reverse of the reward list and multiply by the gamma value
		for r in rewards[::-1]:
			sum_reward = r + self.gamma * sum_reward
			discounted_rewards.append(sum_reward)

		# Revert the list to obtain again the correct order (the computation
		# in the loop was computed reversed)
		discounted_rewards.reverse() 	

		#
		return discounted_rewards

	
	# Computing the loss function of the critic for the gradient descent procedure,
	# learn to predict the expected reward from this state
	def critic_objective_function(self, memory_buffer):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		discounted_reward = np.vstack(memory_buffer[:, 3])

		# The loss function here is a simple gradient descent to train the
		# critic to predict the real value obtained from the environment
		# with a standard MSE loss function
		predicted_value = self.critic(state)
		mse = tf.math.square(predicted_value - discounted_reward)

		#
		return tf.math.reduce_mean(mse)


	# Mandatory method to implement for the ReinforcementLearning class
	# here we select thea action based on the state, for policy gradient method we obtain
	# a probability from the network, from where we perform a sampling
	def get_action(self, state):
		softmax_out = self.actor(state.reshape((1, -1)))
		selected_action = np.random.choice(self.action_space.n, p=softmax_out.numpy()[0])
		return selected_action, softmax_out[0][selected_action]


	# Computing the objective function of the actor for the gradient ascent procedure,
	# here is where the 'magic happens'. 
	# Note that in PPO (both mc and TD) the return is now the advantage instead to the
	# cumulative trajectory reward of REINFORCE. Now it does not make sense to consider 
	# multiple trajectories for the sum and then computing the mean, we can directly consider 
	# a unique trakectories and compute the sum at the final stage.
	# In this context the kind of rollout (mc or TD) does not make any changes, the approach 
	# is still actor critic and the return is a 'single step' advantage.
	def actor_objective_function( self, memory_buffer ):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		action_prob = np.vstack(memory_buffer[:, 2])
		discounted_reward = np.vstack(memory_buffer[:, 3])
		done = np.vstack(memory_buffer[:, 5])

		# Computation of the advantege
		baseline = self.critic(state)
		advantage = discounted_reward - baseline

		#
		prob = self.actor(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)] #Trick to obatin the coordinates of each desired action
		prob = tf.expand_dims(tf.gather_nd(prob, action_idx), axis=-1)
		r_theta = tf.math.divide(prob, action_prob) #prob/old_prob

		#
		clip_val = 0.2
		obj_1 = r_theta * advantage
		obj_2 = tf.clip_by_value(r_theta, 1-clip_val, 1+clip_val) * advantage
		partial_objective = tf.math.minimum(obj_1, obj_2)
		
		#
		return -tf.math.reduce_mean(partial_objective)



class ContMonteCarloPPO( MonteCarloPPO ):

	"""
	Class that inherits from PPO to implements the contonuous
	version of the algorithm with a Monte Carlo approach

	"""


	def __init__( self, env, verbose, str_mod="mcPPO", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed, **kwargs )

		#
		action_norm_bound = [env.action_space.low, env.action_space.high]
		self.actor = self.generate_model(self.input_shape, self.action_space, \
			layers=self.layers, nodes=self.nodes, last_activation='sigmoid', output_bounds=action_norm_bound)

		#
		self.sigma = 1.0
		self.sigma_decay = 0.999999

		#
		self.relevant_params['sigma_decay'] = 'sd'

	
	def get_action(self, state):
		mu = self.actor(state.reshape((1, -1)))
		action = np.random.normal(loc=mu, scale=self.sigma)
		self.sigma *= self.sigma_decay
		return action[0], mu[0]


	def actor_objective_function(self, memory_buffer):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = np.vstack(memory_buffer[:, 1])
		mu = np.vstack(memory_buffer[:, 2])
		discounted_reward = np.vstack(memory_buffer[:, 3])
		done = np.vstack(memory_buffer[:, 5])

		baseline = self.critic(state)
		adv = discounted_reward - baseline # Advantage = MC - baseline

		predictions_mu = self.actor(state)

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
