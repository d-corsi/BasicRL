from basic_rl.gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf


class DDPG( ReinforcementLearning ):

	"""
	Class that inherits from ReinforcementLearning to implements the DDPG, the original paper can be found here:
	https://arxiv.org/abs/1509.02971 [1]

	[1] Continuous control with deep reinforcement learning, 
		Lillicrap et al., 
		arXiv preprint arXiv:1509.02971, 2015

	"""


	# Constructor of the class
	def __init__( self, env, verbose, str_mod="DDPG", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )

		#
		critic_input_shape = (self.input_shape[0]+self.action_space.shape[0], )
		action_norm_bound = [env.action_space.low, env.action_space.high]
		
		#
		self.memory_size = 5000
		self.gamma = 0.99
		self.critic_epoch = 40
		self.critic_batch_size = 128
		self.eps_decay = 0.9995
		self.tau = 0.005
		self.layers = 2
		self.nodes = 32
		self.layers_critic = 3
		self.nodes_critic = 64

		# 
		self.relevant_params = {
			'memory_size': 'ms',
			'gamma' : 'gamma',
			'tau' : 'tau',
			'critic_epoch' : 'ce',
			'critic_batch_size' : 'cbs',
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
		self.actor = self.generate_model(self.input_shape, self.action_space.shape[0], \
			layers=self.layers, nodes=self.nodes,
			last_activation='sigmoid', output_bounds=action_norm_bound)
		
		#
		self.critic = self.generate_model(critic_input_shape, 1, layers=self.layers_critic, nodes=self.nodes_critic)
		self.critic_target = self.generate_model(critic_input_shape, 1, layers=self.layers_critic, nodes=self.nodes_critic)
		self.critic_target.set_weights(self.critic.get_weights())

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()
		self.critic_optimizer = tf.keras.optimizers.Adam()


	# Mandatory method to implement for the ReinforcementLearning class, decide the 
	# update frequency and some variable update for on/off policy algorithms
	# (i.e., eps_greedy, buffer, ...)
	def network_update_rule( self, episode, terminal ):

		# Update of the networks for DQN!
		# Update of the eps greedy strategy
		# - After each episode of the training
		if terminal: 
			self.update_networks( np.array(self.memory_buffer, dtype=object) )
			self.eps_greedy *= self.eps_decay
			self.eps_greedy = max(0.05, self.eps_greedy)

		# Update toward the target network
		self._update_target(self.critic.variables, self.critic_target.variables, tau=self.tau)


	# Application of the gradient with TensorFlow and based on the objective function
	def update_networks( self, memory_buffer ):

		# Critic update (repeated critic_epoch times on a batch, fixed):
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


	# Soft update of the network toward the target network,
	# see the original paper for details
	def _update_target(self, weights, target_weights, tau):
		for (a, b) in zip(target_weights, weights):
			a.assign(b * tau + a * (1 - tau))


	# Mandatory method to implement for the ReinforcementLearning class
	# here we select thea action based on the state, for eps greedy based policy
	# we eprform the exploration selecting random action with a decreasing frequency
	def get_action(self, state):

		action = self.actor(state.reshape((1, -1))).numpy()[0]
		action += np.random.normal(loc=0, scale=self.eps_greedy)

		return action, None


	# Computing the loss function of the critic for the gradient descent procedure,
	# learn to predict the temporal difference formula for the reward
	# here it applies the modified/adapted Bellman equation for DDPG 
	def critic_objective_function( self, memory_buffer ):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = np.vstack(memory_buffer[:, 1])
		reward  = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		# TODO: translate me!
		# Compute the objective function
		# => r + gamma * max(Q(s', a'))
		# L'idea sarebbe quella di utilizzare la Bellman, in particolare vorrei avere il q value dell'azione col valore più alto a pArteire dallo stato s'
		# Trovandomi però con un action sapce continuo non posso testare le diverse azioni
		# Prendo quindi come azione migliore quella che la policy avrebbe scelto in quello stato ,e suppongo sarebbe la scelta migliore
		best_action = self.actor(new_state)
		critic_input = np.hstack([new_state, best_action])
		max_q = self.critic_target(critic_input)

		#
		target = reward + self.gamma * max_q * (1 - done.astype(int))

		#
		critic_input = np.hstack([state, action])
		predicted_values = self.critic(critic_input)
		mse = tf.math.square(predicted_values - target.numpy())
		
		#
		return tf.math.reduce_mean(mse)


	# Computing the objective function of the actor for the gradient ascent procedure,
	# here is where the 'magic happens'
	def actor_objective_function( self, memory_buffer ):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = self.actor(state)	

		#
		target_input = tf.experimental.numpy.hstack([state, action])
		target = self.critic( target_input )

		#
		return -tf.math.reduce_mean(target)
		