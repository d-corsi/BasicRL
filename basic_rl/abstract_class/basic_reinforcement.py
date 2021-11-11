import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow import keras
from collections import deque
import tensorflow as tf
import numpy as np
import abc, random

class BasicReinforcement( metaclass=abc.ABCMeta ):

	train_critic = False
	temporal_difference_method = False
	use_batch = False
	dqn_update = False
	ddpg_update = False
	td3_update = False

	def __init__( self, env ):

		# Configurations
		self.name = "BasicReinforcement"
		self.env = env
		self.verbose = 1
		self.render = False
		self.num_episodes = 1000

		# Network Definitions
		self.input_shape = self.env.observation_space.shape

		self.actor_network = self.get_actor_model()
		self.actor_target = self.get_actor_model()
		self.actor_target.set_weights(self.actor_network.get_weights())
		self.actor_optimizer = keras.optimizers.Adam()

		self.critic_network = self.get_critic_model()
		self.critic_target = self.get_critic_model()
		self.critic_target.set_weights(self.critic_network.get_weights())
		self.critic_optimizer = keras.optimizers.Adam()

		self.critic_network_2 = self.get_critic_model()
		self.critic_target_2 = self.get_critic_model()
		self.critic_target_2.set_weights(self.critic_network_2.get_weights())
		self.critic_optimizer_2 = tf.keras.optimizers.Adam()

		# Training Hyperparameters
		self.gamma = 0.99
		self.sigma = 1.0
		self.sigma_decay = 0.99
		self.update_frequency = 10
		self.batch_size = 64
		self.epoch = 10
		self.memory_size = None
		self.exploration_rate = 1.0
		self.exploration_decay = 0.999
		self.tau = 0.005
		self.td3_noise_clip = 0.2
		self.actor_update_delay = 5

		# Training ID
		self.run_id = np.random.randint(0, 1000)


	def learn( self ):
		reward_list = []
		ep_reward_mean = deque( maxlen=100 )
		memory_buffer = deque( maxlen=self.memory_size )

		for episode in range( self.num_episodes ):
			state = self.env.reset()
			ep_reward = 0
			step_counter = 0

			while True:
				if self.render: self.env.render()
				step_counter += 1
				action, action_prob = self.get_action(state)
				new_state, reward, done, _ = self.env.step(action)
				ep_reward += reward

				memory_buffer.append([state, action, action_prob, reward, new_state, done])
				if done: break
				state = new_state

				if self.dqn_update:
					self.update_networks(memory_buffer)	
					self.update_target_network(self.actor_network.variables, self.actor_target.variables, tau=self.tau)
				if self.ddpg_update:
					self.update_networks(memory_buffer)
					self.update_target_network(self.critic_network.variables, self.critic_target.variables, tau=self.tau)
				if self.td3_update:
					self.update_networks(memory_buffer, only_critic=True)
					if (step_counter % self.actor_update_delay) == 0: self.update_networks(memory_buffer, only_actor=True) #TD3 trick 2
					self.update_target_network(self.critic_network.variables, self.critic_target.variables, tau=self.tau)
					self.update_target_network(self.critic_network_2.variables, self.critic_target_2.variables, tau=self.tau)


			if not (self.dqn_update or self.ddpg_update or self.td3_update) and (episode % self.update_frequency == 0):
				self.update_networks(np.array(memory_buffer, dtype=object))
				memory_buffer.clear()
				self.sigma = self.sigma * self.sigma_decay if self.sigma > 0.05 else 0.05

			self.exploration_rate = self.exploration_rate * self.exploration_decay if self.exploration_rate > 0.05 else 0.05
			
			ep_reward_mean.append(ep_reward)
			reward_list.append(ep_reward)
			self.plot_results( episode, ep_reward, ep_reward_mean, reward_list )


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


	def temporal_difference(self, reward, new_state, done): 
		return reward + (1 - done.astype(int)) * self.gamma * self.critic_network(new_state) # 1-Step TD, for the n-Step TD we must save more sequence in the buffer


	def update_target_network(self, weights, target_weights, tau):
		for (a, b) in zip(target_weights, weights):
			a.assign(b * tau + a * (1 - tau))


	def update_networks(self, memory_buffer, only_actor=False, only_critic=False):
		
		if not self.temporal_difference_method and not self.dqn_update and not self.ddpg_update and not self.td3_update:
			counter = 0
			for i in range(len(memory_buffer)):
				if(memory_buffer[:, 5][i]): 
					memory_buffer[:, 3][counter:i+1] = self.discount_reward(memory_buffer[:, 3][counter:i+1])
					counter = i

		if self.use_batch:
			inner_batch_size = min(len(memory_buffer), self.batch_size)
			mini_batch_n = int(len(memory_buffer) / inner_batch_size)
			batch_list = np.array_split(memory_buffer, mini_batch_n)
			epoch = self.epoch
		elif self.dqn_update or self.ddpg_update or self.td3_update:
			samples = np.array(random.sample(memory_buffer, min(len(memory_buffer), self.batch_size)), dtype=object)
			epoch = 1
			batch_list = np.array([samples])
		else:
			epoch = 1
			batch_list = np.array([memory_buffer])


		for _ in range(epoch):
			for current_batch in batch_list:
				with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c, tf.GradientTape() as tape_c2:
					if not only_critic:
						objective_function_a = self.actor_objective_function(current_batch) #Compute loss with custom loss function
						grads_a = tape_a.gradient(objective_function_a, self.actor_network.trainable_variables) #Compute gradients actor for network
						self.actor_optimizer.apply_gradients( zip(grads_a, self.actor_network.trainable_variables) ) #Apply gradients to update network weights
					
					if self.train_critic and not only_actor:
						objective_function_c = self.critic_objective_function(current_batch) #Compute loss with custom loss function
						grads_c = tape_c.gradient(objective_function_c, self.critic_network.trainable_variables) #Compute gradients critic for networks
						self.critic_optimizer.apply_gradients( zip(grads_c, self.critic_network.trainable_variables) ) #Apply gradients to update network weights

						if self.td3_update:
							objective_function_c2 = self.critic_objective_function(current_batch, self.critic_network_2) #Compute loss with custom loss function
							grads_c2 = tape_c2.gradient(objective_function_c2, self.critic_network_2.trainable_variables) #Compute gradients critic for networks
							self.critic_optimizer_2.apply_gradients( zip(grads_c2, self.critic_network_2.trainable_variables) ) #Apply gradients to update network weights


			if self.use_batch: random.shuffle(batch_list)


	def critic_objective_function(self, memory_buffer):
		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		predicted_value = self.critic_network(state)
		if not self.temporal_difference_method: target = reward 
		else: target = self.temporal_difference(reward, new_state, done) 
		mse = tf.math.square(predicted_value - target)

		return tf.math.reduce_mean(mse)


	def get_critic_model( self ):
		inputs = keras.layers.Input(shape=self.input_shape)
		hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(1, activation='linear')(hidden_1)

		return keras.Model(inputs, outputs)


	@abc.abstractmethod
	def plot_results( self, episode, ep_reward, ep_reward_mean, reward_list ): return

	@abc.abstractmethod
	def get_action( self, state ): return

	@abc.abstractmethod
	def actor_objective_function( self, memory_buffer ): return

	@abc.abstractmethod
	def get_actor_model( self ): return
