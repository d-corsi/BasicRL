from tensorflow import keras
from collections import deque
import tensorflow as tf
import numpy as np
import gym

class REINFORCE:
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

		self.optimizer = keras.optimizers.Adam()
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

				memory_buffer.append([state, reward, action])
				if done: break
				state = new_state

			self.update_networks(np.array(memory_buffer, dtype=object))
			memory_buffer.clear()
			self.sigma = self.sigma * self.exploration_decay if self.sigma > 0.05 else 0.05
			
			ep_reward_mean.append(ep_reward)
			reward_list.append(ep_reward)
			if self.verbose > 0 and not self.discrete: print(f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, sigma: {self.sigma:0.2f}")
			if self.verbose > 0 and self.discrete: print(f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}") 
			if self.verbose > 1: np.savetxt(f"data/reward_REINFORCE_{self.run_id}.txt", reward_list)
			

	def update_networks(self, memory_buffer):
		memory_buffer[:, 1] = self.discount_reward(memory_buffer[:, 1]) # Discount the rewards in a MC way
		
		with tf.GradientTape() as tape:
			objective_function = self.actor_objective_function(memory_buffer) #Compute loss with custom loss function
			grads = tape.gradient(objective_function, self.actor.trainable_variables) #Compute gradients actor for network
			self.optimizer.apply_gradients( zip(grads, self.actor.trainable_variables) ) #Apply gradients to update network weights

	
	def discount_reward(self, rewards):
		sum_reward = 0
		discounted_rewards = []

		for r in rewards[::-1]:
			sum_reward = r + self.gamma * sum_reward
			discounted_rewards.append(sum_reward)
		discounted_rewards.reverse() 

		# Normalize
		eps = np.finfo(np.float64).eps.item()  # Smallest number such that 1.0 + eps != 1.0 
		discounted_rewards = np.array(discounted_rewards)
		discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)

		return discounted_rewards


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
		reward = np.vstack(memory_buffer[:, 1])
		action = memory_buffer[:, 2]

		baseline = np.mean(reward)
		probability = self.actor(state)
		action_idx = [[counter, val] for counter, val in enumerate(action)]
		probability = tf.expand_dims(tf.gather_nd(probability, action_idx), axis=-1)
		partial_objective = tf.math.log(probability) * (reward - baseline)

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
		reward = np.vstack(memory_buffer[:, 1])
		action = np.vstack(memory_buffer[:, 2])

		baseline = np.mean(reward)
		mu = self.actor(state)
		pdf_value = tf.sqrt(1/(2 * np.pi * self.sigma**2)) * tf.exp(-(action - mu)**2/(2 * self.sigma**2))
		pdf_value = tf.math.reduce_mean(pdf_value, axis=1, keepdims=True)
		partial_objective = tf.math.log(pdf_value) * (reward - baseline)

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



	