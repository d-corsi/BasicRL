from tkinter.messagebox import NO
from basic_rl.gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf


class DDQN( ReinforcementLearning ):

	def __init__( self, env, verbose, str_mod="DDQN", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )

		#
		self.network = self.generate_model(self.input_shape, self.action_space.n)
		self.network_target = self.generate_model(self.input_shape, self.action_space.n)
		self.network_target.set_weights(self.network.get_weights())

		#
		self.optimizer = tf.keras.optimizers.Adam()
		
		#
		self.memory_size = 5000
		self.gamma = 0.99
		self.epoch = 40
		self.batch_size = 128
		self.eps_decay = 0.9995
		self.tau = 0.005

		# 
		self.relevant_params = {
			'memory_size': 'ms',
			'gamma' : 'gamma',
			'tau' : 'tau',
			'epoch' : 'epoch',
			'batch_size' : 'bs',
			'eps_decay' : 'decay'
		}

		# Override the default parameters with kwargs
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		#
		self.memory_buffer = deque( maxlen=self.memory_size )
		self.eps_greedy = 1


	def network_update_rule( self, episode, terminal ):

		# Update of the networks for DQN!
		# Update of the eps greedy strategy
		# - After each episode of the training
		if terminal: 
			self.update_networks(np.array(self.memory_buffer, dtype=object))
			self.eps_greedy *= self.eps_decay
			self.eps_greedy = max(0.05, self.eps_greedy)

		# Update toward the target network
		self._update_target(self.network.variables, self.network_target.variables, tau=self.tau)


	def update_networks( self, memory_buffer ):

		#
		for _ in range( self.epoch ):
			idx = np.random.randint(memory_buffer.shape[0], size=self.batch_size)
			training_batch = memory_buffer[idx]
			with tf.GradientTape() as tape:
				objective_function = self.objective_function( training_batch )
				gradient = tape.gradient( objective_function, self.network.trainable_variables )
				self.optimizer.apply_gradients( zip(gradient, self.network.trainable_variables) )


	def _update_target(self, weights, target_weights, tau):
		for (a, b) in zip(target_weights, weights):
			a.assign(b * tau + a * (1 - tau))


	def get_action(self, state):

		if np.random.random() < self.eps_greedy:
			action = np.random.choice(self.action_space)
		else:
			action = np.argmax(self.network(state.reshape((1, -1))))

		return action, 0


	def objective_function( self, memory_buffer ):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		reward  = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		next_state_action = np.argmax(self.network(new_state), axis=1)
		target_mask = self.network_target(new_state) * tf.one_hot(next_state_action, self.action_space)
		target_mask = tf.reduce_sum(target_mask, axis=1, keepdims=True)
		
		target_value = reward + (1 - done.astype(int)) * self.gamma * target_mask
		mask = self.network(state) * tf.one_hot(action, self.action_space)
		prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

		mse = tf.math.square(prediction_value - target_value)
		
		return tf.math.reduce_mean(mse)
