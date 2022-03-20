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
		self.actor = self.generate_model(self.input_shape, self.action_space.n, last_activation='softmax')
		self.critic = self.generate_model(self.input_shape)

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()
		self.critic_optimizer = tf.keras.optimizers.Adam()

		#
		self.memory_size = None
		self.gamma = 0.99
		self.trajectory_update = 10
		self.critic_epoch = 40
		self.critic_batch_size = 128

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

		self.memory_buffer = deque( maxlen=self.memory_size )


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

		# Replace the reward with the discoutned returs in the buffer
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
	# here is where the 'magic happens'
	def actor_objective_function( self, memory_buffer ):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		action_prob = np.vstack(memory_buffer[:, 2])
		discounted_reward  = np.vstack(memory_buffer[:, 3])
		done = np.vstack(memory_buffer[:, 5])

		end_trajectories = np.where(done == True)[0]

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
		trajectory_probabilities = []; counter = 0
		for i in end_trajectories:
			trajectory_probabilities.append( tf.math.reduce_sum( tf.math.log(partial_objective[counter : i+1])) )
			counter = i+1

		#
		return -tf.math.reduce_mean(partial_objective)
