from basic_rl.gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf


class DDQN( ReinforcementLearning ):

	"""
	Class that inherits from ReinforcementLearning to implements the Double DQN algorithm, the original paper can be found here:
	https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf [1] and https://ojs.aaai.org/index.php/AAAI/article/view/10295 [2]

	[1] Playing atari with deep reinforcement learning, 
		Mnih et al., 
		arXiv preprint arXiv:1312.5602, 2013

	[1] Playing atari with deep reinforcement learning, 
		Van Hasselt et al., 
		Proceedings of the AAAI conference on artificial intelligence, 2016

	"""


	# Constructor of the class
	def __init__( self, env, verbose, str_mod="DDQN", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )
		
		#
		self.memory_size = 5000
		self.gamma = 0.99
		self.epoch = 40
		self.batch_size = 128
		self.eps_decay = 0.9995
		self.tau = 0.005
		self.layers = 2
		self.nodes = 32

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

		#
		self.network = self.generate_model(self.input_shape, self.action_space.n, \
			layers=self.layers, nodes=self.nodes)
		self.network_target = self.generate_model(self.input_shape, self.action_space.n, \
			layers=self.layers, nodes=self.nodes)
		self.network_target.set_weights(self.network.get_weights())

		#
		self.optimizer = tf.keras.optimizers.Adam()


	# Mandatory method to implement for the ReinforcementLearning class, decide the 
	# update frequency and some variable update for on/off policy algorithms
	# (i.e., eps_greedy, buffer, ...)
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


	# Application of the gradient with TensorFlow and based on the objective function
	def update_networks( self, memory_buffer ):

		#
		for _ in range( self.epoch ):

			# Computing a random sample of elements from the batch for the training,
			# randomized at each epoch
			idx = np.random.randint(memory_buffer.shape[0], size=self.batch_size)
			training_batch = memory_buffer[idx]

			#
			with tf.GradientTape() as tape:

				# Compute the objective function, compute the gradient information and apply the
				# gradient with the optimizer
				objective_function = self.objective_function( training_batch )
				gradient = tape.gradient( objective_function, self.network.trainable_variables )
				self.optimizer.apply_gradients( zip(gradient, self.network.trainable_variables) )


	# Soft update of the network toward the target network,
	# see the original paper for details
	def _update_target(self, weights, target_weights, tau):
		for (a, b) in zip(target_weights, weights):
			a.assign(b * tau + a * (1 - tau))



	# Mandatory method to implement for the ReinforcementLearning class
	# here we select thea action based on the state, for eps greedy based policy
	# we eprform the exploration selecting random action with a decreasing frequency
	def get_action(self, state):

		if np.random.random() < self.eps_greedy:
			action = np.random.choice(self.action_space.n)
		else:
			action = np.argmax(self.network(state.reshape((1, -1))))

		return action, 0


	# Computing the objective function of the DQN for the gradient descent procedure,
	# here it applies the Bellman equation
	def objective_function( self, memory_buffer ):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		reward  = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		# 
		next_state_action = np.argmax(self.network(new_state), axis=1)
		target_mask = self.network_target(new_state) * tf.one_hot(next_state_action, self.action_space.n)
		target_mask = tf.reduce_sum(target_mask, axis=1, keepdims=True)
		
		#
		target_value = reward + (1 - done.astype(int)) * self.gamma * target_mask
		mask = self.network(state) * tf.one_hot(action, self.action_space.n)
		prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

		#
		mse = tf.math.square(prediction_value - target_value)
		
		#
		return tf.math.reduce_mean(mse)
