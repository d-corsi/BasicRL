from basic_rl.abstract_class.basic_reinforcement import BasicReinforcement
import numpy as np
import tensorflow as tf

class DDQN( BasicReinforcement ):

	dqn_update = True

	def __init__(self, env, **kwargs):
		self.action_space = env.action_space.n

		super().__init__(env)

		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		self.memory_size = 2000


	def plot_results( self, episode, ep_reward, ep_reward_mean, reward_list ):
			if self.verbose > 0: 
				print(f"{self.name} Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, exploration: {self.exploration_rate:0.2f}")
			if self.verbose > 1: 
				np.savetxt(f"data/reward_{self.name}_{self.run_id}.txt", reward_list)


	def get_action(self, state):
		if np.random.random() < self.exploration_rate:
			action = np.random.choice(self.action_space)
		else:
			action = np.argmax(self.actor_network(state.reshape((1, -1)))) 
		return action, None

	
	def actor_objective_function(self, replay_buffer):
		state = np.vstack(replay_buffer[:, 0])
		action = replay_buffer[:, 1]
		reward = np.vstack(replay_buffer[:, 3])
		new_state = np.vstack(replay_buffer[:, 4])
		done = np.vstack(replay_buffer[:, 5])

		next_state_action = np.argmax(self.actor_network(new_state), axis=1)
		target_mask = self.actor_target(new_state) * tf.one_hot(next_state_action, self.action_space)
		target_mask = tf.reduce_sum(target_mask, axis=1, keepdims=True)
		
		target_value = reward + (1 - done.astype(int)) * self.gamma * target_mask
		mask = self.actor_network(state) * tf.one_hot(action, self.action_space)
		prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

		mse = tf.math.square(prediction_value - target_value)
		return tf.math.reduce_mean(mse)


	def get_actor_model( self ):
		inputs = tf.keras.layers.Input(shape=self.input_shape)
		hidden_0 = tf.keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = tf.keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = tf.keras.layers.Dense(self.action_space, activation='linear')(hidden_1)

		return tf.keras.Model(inputs, outputs)


class DQN( DDQN ):
	
	def actor_objective(self, replay_buffer):
		state = np.vstack(replay_buffer[:, 0])
		action = replay_buffer[:, 1]
		reward = np.vstack(replay_buffer[:, 3])
		new_state = np.vstack(replay_buffer[:, 4])
		done = np.vstack(replay_buffer[:, 5])

		target_value = reward + (1 - done.astype(int)) * self.gamma * np.amax(self.actor_network(new_state), axis=1, keepdims=True)
		mask = self.actor_network(state) * tf.one_hot(action, self.action_space)
		prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

		mse = tf.math.square(prediction_value - target_value)
		return tf.math.reduce_mean(mse)


class TargetDQN( DDQN ):
	
	def actor_objective_function(self, replay_buffer):
		state = np.vstack(replay_buffer[:, 0])
		action = replay_buffer[:, 1]
		reward = np.vstack(replay_buffer[:, 3])
		new_state = np.vstack(replay_buffer[:, 4])
		done = np.vstack(replay_buffer[:, 5])

		target_value = reward + (1 - done.astype(int)) * self.gamma * np.amax(self.actor_target(new_state), axis=1, keepdims=True)
		mask = self.actor_network(state) * tf.one_hot(action, self.action_space)
		prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

		mse = tf.math.square(prediction_value - target_value)
		return tf.math.reduce_mean(mse)