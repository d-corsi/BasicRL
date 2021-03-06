from tensorflow import keras
from collections import deque
import tensorflow as tf
import numpy as np
import gym

class A2C:
	def __init__(self, env, discrete, verbose):
		self.env = env
		self.discrete = discrete
		self.verbose = verbose

		self.input_shape = self.env.observation_space.shape
		if(self.discrete): self.action_space = env.action_space.n
		else: self.action_space = env.action_space.shape[0]

		if(self.discrete): 
			self.actor = self.get_actor_model_disc(self.input_shape, self.action_space)
			self.get_action = self.get_action_disc
			self.actor_objective_function = self.actor_objective_function_disc
		else: 
			self.actor = self.get_actor_model_cont(self.input_shape, self.action_space, [env.action_space.low, env.action_space.high])
			self.get_action = self.get_action_cont
			self.actor_objective_function = self.actor_objective_function_cont

		self.critic = self.get_critic_model(self.input_shape)

		self.actor_optimizer = keras.optimizers.Adam()
		self.critic_optimizer = keras.optimizers.Adam()
		self.gamma = 0.99
		self.sigma = 1.0
		self.exploration_decay = 1	

		self.run_id = np.random.randint(0, 1000)
		self.render = False


	def loop( self, num_episodes=1000 ):
		reward_list = []
		ep_reward_mean = deque(maxlen=100)
		memory_buffer = deque()

		for episode in range(num_episodes):
			state = self.env.reset()
			ep_reward = 0

			while True:
				if self.render: self.env.render()
				action = self.get_action(state)
				new_state, reward, done, _ = self.env.step(action)
				ep_reward += reward

				memory_buffer.append([state, action, reward, new_state, done])
				if done: break
				state = new_state

			self.update_networks(np.array(memory_buffer, dtype=object))
			memory_buffer.clear()
			self.sigma = self.sigma * self.exploration_decay if self.sigma > 0.05 else 0.05
			
			ep_reward_mean.append(ep_reward)
			reward_list.append(ep_reward)
			if self.verbose > 0 and not self.discrete: print(f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, sigma: {self.sigma:0.2f}")
			if self.verbose > 0 and self.discrete: print(f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}") 
			if self.verbose > 1: np.savetxt(f"data/reward_A2C_{self.run_id}.txt", reward_list)


	def update_networks(self, memory_buffer):
		with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
			objective_function_c = self.critic_objective_function(memory_buffer) #Compute loss with custom loss function
			objective_function_a = self.actor_objective_function(memory_buffer) #Compute loss with custom loss function

			grads_c = tape_c.gradient(objective_function_c, self.critic.trainable_variables) #Compute gradients critic for network
			grads_a = tape_a.gradient(objective_function_a, self.actor.trainable_variables) #Compute gradients actor for network

			self.critic_optimizer.apply_gradients( zip(grads_c, self.critic.trainable_variables) ) #Apply gradients to update network weights
			self.actor_optimizer.apply_gradients( zip(grads_a, self.actor.trainable_variables) ) #Apply gradients to update network weights


	def _Gt(self, reward, new_state, done): 
		return reward + (1 - done.astype(int)) * self.gamma * self.critic(new_state) # 1-Step TD, for the n-Step TD we must save more sequence in the buffer


	##########################
    ##### CRITIC METHODS #####
    ##########################


	def get_critic_model(self, input_shape):
		inputs = keras.layers.Input(shape=input_shape)
		hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(1, activation='linear')(hidden_1)

		return keras.Model(inputs, outputs)

	
	def critic_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		reward = np.vstack(memory_buffer[:, 2])
		new_state = np.vstack(memory_buffer[:, 3])
		done = np.vstack(memory_buffer[:, 4])

		predicted_value = self.critic(state)
		target = self._Gt(reward, new_state, done)
		mse = tf.math.square(predicted_value - target)

		return tf.math.reduce_mean(mse)

	
	##########################
    #### DISCRETE METHODS ####
    ##########################


	def get_action_disc(self, state):
		softmax_out = self.actor(state.reshape((1, -1)))
		selected_action = np.random.choice(self.action_space, p=softmax_out.numpy()[0])
		return selected_action


	def actor_objective_function_disc(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		reward = np.vstack(memory_buffer[:, 2])
		new_state = np.vstack(memory_buffer[:, 3])
		done = np.vstack(memory_buffer[:, 4])

		baseline = self.critic(state)
		probability = self.actor(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)]
		probability = tf.expand_dims(tf.gather_nd(probability, action_idx), axis=-1)
		partial_objective = tf.math.log(probability) * (self._Gt(reward, new_state, done) - baseline)

		return -tf.math.reduce_mean(partial_objective)
	
		
	def get_actor_model_disc(self, input_shape, output_size):
		inputs = keras.layers.Input(shape=input_shape)
		hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(output_size, activation='softmax')(hidden_1)

		return keras.Model(inputs, outputs)


   	##########################
    ### CONTINUOUS METHODS ###
    ##########################	


	def get_action_cont(self, state):
		mu = self.actor(state.reshape((1, -1)))
		action = np.random.normal(loc=mu, scale=self.sigma)
		return action[0]


	def actor_objective_function_cont(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = np.vstack(memory_buffer[:, 1])
		reward = np.vstack(memory_buffer[:, 2])
		new_state = np.vstack(memory_buffer[:, 3])
		done = np.vstack(memory_buffer[:, 4])

		baseline = self.critic(state)
		mu = self.actor(state)
		pdf_value = tf.sqrt(1/(2 * np.pi * self.sigma**2)) * tf.exp(-(action - mu)**2/(2 * self.sigma**2))
		pdf_value = tf.math.reduce_mean(pdf_value, axis=1, keepdims=True)
		partial_objective = tf.math.log(pdf_value) * (self._Gt(reward, new_state, done) - baseline)

		return -tf.math.reduce_mean(partial_objective)


	def get_actor_model_cont(self, input_shape, output_size, output_range):
		# Initialize weights between -3e-3 and 3-e3
		last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		inputs = keras.layers.Input(shape=input_shape)
		hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(output_size, activation='sigmoid', kernel_initializer=last_init)(hidden_1)

		# Fix output range with the range of the action
		outputs = outputs * (output_range[1] - output_range[0]) + output_range[0]

		return keras.Model(inputs, outputs)
