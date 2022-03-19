from tkinter.messagebox import NO
from basic_rl.gym_loop import ReinforcementLearning
from collections import deque
import numpy as np
import tensorflow as tf


class DDPG( ReinforcementLearning ):

	def __init__( self, env, verbose, str_mod="DDPG", seed=None, **kwargs ):

		#
		super().__init__( env, verbose, str_mod, seed )

		#
		tf.random.set_seed( seed )
		np.random.seed( seed )

		#
		critic_input_shape = (self.input_shape[0]+self.action_space.shape[0], )
		self.actor = self.generate_model(self.input_shape, self.action_space.shape[0])
		self.critic = self.generate_model(critic_input_shape, 1, layers=3, nodes=64)

		#
		self.critic_target = self.generate_model(critic_input_shape, 1, layers=3, nodes=64)
		self.critic_target.set_weights(self.critic.get_weights())

		#
		self.actor_optimizer = tf.keras.optimizers.Adam()
		self.critic_optimizer = tf.keras.optimizers.Adam()
		
		#
		self.memory_size = 5000
		self.gamma = 0.99
		self.critic_epoch = 40
		self.critic_batch_size = 128
		self.eps_decay = 0.9995
		self.tau = 0.005

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


	def update_networks( self, memory_buffer ):

		# Critic update (repeated critic_epoch times on a batch, fixed):
		for _ in range( self.critic_epoch ):
			idx = np.random.randint(memory_buffer.shape[0], size=self.critic_batch_size)
			training_batch = memory_buffer[idx]
			with tf.GradientTape() as critic_tape:
				critic_objective_function = self.critic_objective_function( training_batch )
				critic_gradient = critic_tape.gradient( critic_objective_function, self.critic.trainable_variables )
				self.critic_optimizer.apply_gradients( zip(critic_gradient, self.critic.trainable_variables) )

		# Actor update:
		with tf.GradientTape() as actor_tape:
			actor_objective_function = self.actor_objective_function( memory_buffer )
			actor_gradient = actor_tape.gradient(actor_objective_function, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients( zip(actor_gradient, self.actor.trainable_variables) )


	def _update_target(self, weights, target_weights, tau):
		for (a, b) in zip(target_weights, weights):
			a.assign(b * tau + a * (1 - tau))


	def get_action(self, state):

		action = self.actor(state.reshape((1, -1))).numpy()[0]
		action += np.random.normal(loc=0, scale=self.eps_greedy)

		return action, None


	def critic_objective_function( self, memory_buffer ):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = np.vstack(memory_buffer[:, 1])
		reward  = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		# Compute the objective function
		# => r + gamma * max(Q(s', a'))
		# L'idea sarebbe quella di utilizzare la Bellman, in particolare vorrei avere il q value dell'azione col valore più alto a pArteire dallo stato s'
		# Trovandomi però con un action sapce continuo non posso testare le diverse azioni
		# Prendo quindi come azione migliore quella che la policy avrebbe scelto in quello stato ,e suppongo sarebbe la scelta migliore
		best_action = self.actor(new_state)
		critic_input = np.hstack([new_state, best_action])
		max_q = self.critic_target(critic_input)

		target = reward + self.gamma * max_q * (1 - done.astype(int))

		critic_input = np.hstack([state, action])
		predicted_values = self.critic(critic_input)
		mse = tf.math.square(predicted_values - target.numpy())
		
		return tf.math.reduce_mean(mse)


	def actor_objective_function( self, memory_buffer ):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = self.actor(state)	

		target_input = tf.experimental.numpy.hstack([state, action])
		target = self.critic( target_input )

		return -tf.math.reduce_mean(target)