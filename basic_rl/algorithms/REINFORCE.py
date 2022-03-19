from basic_rl.gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf


class Reinforce( ReinforcementLearning ):

	def __init__( self, env, verbose, str_mod="REINFORCE", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )

		#
		self.actor = self.generate_model(self.input_shape, self.action_space.n, last_activation='softmax')

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()

		#
		self.memory_size = None
		self.gamma = 0.99
		self.trajectory_update = 10
		self.trajectory_mean = False

		# 
		self.relevant_params = {
			'gamma' : 'gamma',
			'trajectory_update' : 'tu'
		}

		# Override the default parameters with kwargs
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		self.memory_buffer = deque( maxlen=self.memory_size )


	def network_update_rule( self, episode, terminal ):

		# Update of the networks for Reinforce!
		# - Performed every <trajectory_update> episode
		# - clean up of the buffer at each update (on-policy).
		# - Only on terminal state (collect the full trajectory)
		if terminal and episode % self.trajectory_update == 0:
			self.update_networks(np.array(self.memory_buffer, dtype=object))
			self.memory_buffer.clear()


	def update_networks( self, memory_buffer ):

		# Actor update (repeated #epoch times):
		with tf.GradientTape() as actor_tape:
			actor_objective_function = self.actor_objective_function( memory_buffer )
			actor_gradient = actor_tape.gradient(actor_objective_function, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients( zip(actor_gradient, self.actor.trainable_variables) )


	def get_action(self, state):
		softmax_out = self.actor(state.reshape((1, -1)))
		selected_action = np.random.choice(self.action_space.n, p=softmax_out.numpy()[0])
		return selected_action, None

	
	def actor_objective_function( self, memory_buffer ):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		reward = memory_buffer[:, 3]
		action = memory_buffer[:, 1]
		done = np.vstack(memory_buffer[:, 5])

		end_trajectories = np.where(done == True)[0]
		
		probability = self.actor(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)]
		probability = tf.expand_dims(tf.gather_nd(probability, action_idx), axis=-1)

		trajectory_probabilities = []
		trajectory_rewards = []
		counter = 0
		for i in end_trajectories:
			trajectory_probabilities.append( tf.math.reduce_sum( tf.math.log(probability[counter : i+1])) )
			trajectory_rewards.append( sum(reward[counter : i+1]) )
			counter = i+1

		trajectory_objectives = []
		for log_prob, rw in zip(trajectory_probabilities, trajectory_rewards):
			trajectory_objectives.append( log_prob * (rw - np.mean(trajectory_rewards)) )

		objective_function = tf.reduce_mean( trajectory_objectives )
		return -objective_function
	