from basic_rl.abstract_class.basic_reinforcement import BasicReinforcement
import numpy as np
import tensorflow as tf

class DDPG( BasicReinforcement ):

	train_critic = True
	ddpg_update = True

	def __init__(self, env, **kwargs):
		self.action_space = env.action_space.shape
		self.output_range = [env.action_space.low, env.action_space.high]

		super().__init__(env)

		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		self.memory_size = 5000
		


	def plot_results( self, episode, ep_reward, ep_reward_mean, reward_list ):
		if self.verbose > 0: 
			print(f"{self.name} Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, exploration: {self.exploration_rate:0.2f}")
		if self.verbose > 1: 
			np.savetxt(f"data/reward_{self.name}_{self.run_id}.txt", reward_list)


	def get_action(self, state):
		action = self.actor_network(state.reshape((1, -1))).numpy()[0]
		action += np.random.normal(loc=0, scale=self.exploration_rate)
		return action, None


	def actor_objective_function(self, replay_buffer):
		# Extract values from buffer
		state = np.vstack(replay_buffer[:, 0])
		action = np.vstack(replay_buffer[:, 1])

		action = self.actor_network(state)	
		target = self.critic_network([state, action])

		return -tf.math.reduce_mean(target)

	
	def critic_objective_function(self, replay_buffer):
		# Extract values from buffer
		state = np.vstack(replay_buffer[:, 0])
		action = np.vstack(replay_buffer[:, 1])
		reward = np.vstack(replay_buffer[:, 3])
		new_state = np.vstack(replay_buffer[:, 4])
		done = np.vstack(replay_buffer[:, 5])

		# Compute the objective function
		# => r + gamma * max(Q(s', a'))
		# L'idea sarebbe quella di utilizzare la Bellman, in particolare vorrei avere il q value dell'azione col valore più alto a pArteire dallo stato s'
		# Trovandomi però con un action sapce continuo non posso testare le diverse azioni
		# Prendo quindi come azione migliore quella che la policy avrebbe scelto in quello stato ,e suppongo sarebbe la scelta migliore
		best_action = self.actor_network(new_state)
		max_q = self.critic_target([new_state, best_action])

		target = reward + self.gamma * max_q * (1 - done.astype(int))

		predicted_values = self.critic_network([state, action])
		mse = tf.math.square(predicted_values - target.numpy())

		return tf.math.reduce_mean(mse)


	def get_actor_model( self ):
		# Initialize weights between -3e-3 and 3-e3
		last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		inputs = tf.keras.layers.Input(shape=self.input_shape)
		hidden_0 = tf.keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = tf.keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = tf.keras.layers.Dense(self.action_space[0], activation='sigmoid', kernel_initializer=last_init)(hidden_1)

		# Fix output range with the range of the action
		outputs = outputs * (self.output_range[1] - self.output_range[0]) + self.output_range[0]

		return tf.keras.Model(inputs, outputs)


	def get_critic_model( self ):
		inputs = tf.keras.layers.Input(shape=self.input_shape)
		hidden_inputs = tf.keras.layers.Dense(32, activation='relu')(inputs)

		input_actions = tf.keras.layers.Input(shape=self.action_space)
		hidden_input_action = tf.keras.layers.Dense(32, activation='relu')(input_actions)

		concat = tf.keras.layers.Concatenate()([hidden_inputs, hidden_input_action])

		hidden_0 = tf.keras.layers.Dense(64, activation='relu')(concat)
		hidden_1 = tf.keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = tf.keras.layers.Dense(1, activation='linear')(hidden_1)

		return tf.keras.Model([inputs, input_actions], outputs)
