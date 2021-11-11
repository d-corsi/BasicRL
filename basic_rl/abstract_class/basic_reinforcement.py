import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow import keras
from collections import deque
import tensorflow as tf
import numpy as np
import abc, random

class BasicReinforcement( metaclass=abc.ABCMeta ):

	"""
	Abstract class that implements the basic function shared by all the reinforcment learning algorithm implemented 
	in BasicRL. 
	This class provides also the interface for methods that are different for each implementations andthe hyperparameters 
	of the training for the different algorithms and manteins the default value	for them. 
	This class contains 6 flag that provides little changes based on the selected algorithm for the training:

		- train_critic (used for all the actor/critic approaches, like PPO, A2C or DDPG)
		- temporal_difference_method 
		- use_batch (for the algorithms that make use of batches to train, like PPO )
		- dqn_update (specific when update DQN, the actor must be updated like a critic)
		- ddpg_update (specific when update DDPG, actor and critic have different functions)
		- td3_update (activate the 3 trick of TD3, delayed actor update, target critic, objective with double critic )

	NB: This class is abstract, the default hyperparamters are update by default in the child class

	Methods
	-------
		def learn( )
			method that start the training loop with the given parameters
		update_networks( memory_buffer, only_actor, only_critic )
			method that perform the actual training, compute the objective function, the gradient and perform 
			the backpropagation step
		discount_reward( rewards ):
			method that discount the given reward list using the class parameter self.gamma
		temporal_difference( reward, new_state, done )
			method that compute the temporal difference of 1 step (TD1) from the given 
			parameters list
		update_target_network( fixed_weights, target_weights )
			method that update the weights of the target_weights towards the fixed_weights
		critic_objective_function( memory_buffer )
			method that compute the objective function for the critic, returns the objective function to optimize
		get_critic_model( )
			method that returns the neural network to train for the critic
		def plot_results( episode, ep_reward, ep_reward_mean, reward_list )
			[ABSTRACT] method that select the action given the state, calling the actor network
			or the expolaration policy
		get_action( state )
			[ABSTRACT] method that select the action given the state, calling the actor network
			or the expolaration policy
		actor_objective_function( memory_buffer )
			[ABSTRACT] method that compute the objective function for the actor, returns the objective function to optimize
		get_actor_model( ) 
			[ABSTRACT] method that returns the neural network to train for the actor
	
	
	"""

	train_critic = False
	temporal_difference_method = False
	use_batch = False
	dqn_update = False
	ddpg_update = False
	td3_update = False

	def __init__( self, env ):

		"""
		Constructor of the class

		Parameters
		----------
			env : Gym
				the environment to train, in the Open AI Gym style 
		"""

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
					self.update_target_network(self.critic_network.variables, self.critic_target.variables)
					self.update_target_network(self.critic_network_2.variables, self.critic_target_2.variables)


			if not (self.dqn_update or self.ddpg_update or self.td3_update) and (episode % self.update_frequency == 0):
				self.update_networks(np.array(memory_buffer, dtype=object))
				memory_buffer.clear()
				self.sigma = self.sigma * self.sigma_decay if self.sigma > 0.05 else 0.05

			self.exploration_rate = self.exploration_rate * self.exploration_decay if self.exploration_rate > 0.05 else 0.05
			
			ep_reward_mean.append(ep_reward)
			reward_list.append(ep_reward)
			self.plot_results( episode, ep_reward, ep_reward_mean, reward_list )


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


	def discount_reward(self, rewards):

		"""
		method that discount the given reward list using the class parameter self.gamma

		Parameters
		----------
			rewards : list 
				complete list of the reward to discount

		Returns:
		--------
			discounted_rewards : list
				complete list of the discounted reward
		"""

		# initialize the reward list and the counter for the discoutned future reward
		sum_reward = 0
		discounted_rewards = []

		# iterate over the rewards in the reverse order
		for r in rewards[::-1]:

			# apply the discount in the reverse order and append to the list
			sum_reward = r + self.gamma * sum_reward
			discounted_rewards.append(sum_reward)

		# reverse the order again to restore the correct order
		discounted_rewards.reverse() 

		# Normalize the value, eps is a small number to 
		# eventually avoid division by 0
		eps = np.finfo(np.float64).eps.item()
		discounted_rewards = np.array(discounted_rewards)
		discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)

		#
		return discounted_rewards

	
	def temporal_difference(self, reward, new_state, done): 
		
		"""
		method that compute the temporal difference of 1 step (TD1) from the given parameters list

		Parameters
		----------
			reward : list 
				instant reward
			new_state : NumpyArray
				the next state 
			done: list
				indicate if new_state is terminal with a list of boolean

		Returns:
		--------
			td_1: 1 step temporal difference given the paramters
		"""

		# Compute TD1 using the standard formula
		# refer to "spinning Up" from OpenAI for theoreticl details
		td_1 = reward + (1 - done.astype(int)) * self.gamma * self.critic_network(new_state) 

		#
		return td_1


	def update_target_network(self, fixed_weights, target_weights ):
		"""
		method that update the weights of the target_weights towards the fixed_weights

		Parameters
		----------
			fixed_weights : NumpyArray 
				the weight of the basic network, move the other netowrk weights toward these weights
				with a step of the paramter self.tau (default 0.005 at each step)
			target_weights : NumpyArray
				the netowrk that change its values toward the other network
		"""

		# Iterate over all the weights and update the weights of target_weights in direction of weights
		# the update is of tau towards the "fixed_weights" and (1-tau) remains the same of "target_weights".
		for (a, b) in zip(target_weights, fixed_weights):
			a.assign(b * self.tau + a * (1 - self.tau))


	def critic_objective_function(self, memory_buffer):

		"""
		method that compute the objective function for the critic, returns the objective function to optimize

		Parameters
		----------
			memory_buffer : NumpyArray 
				buffer with the data from each episode, in the following form:
				[state, action, action_prob, reward, new_state, done]

		Returns:
		--------
			objective_function : tf.EagerTensor
				tensorflow tensor that represent the objective function to maximize
		"""

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		reward = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		# Compute the objective function for the critic using mse
		# both temporal difference and montecarlo metod for 
		# the objective
		predicted_value = self.critic_network(state)
		if not self.temporal_difference_method: target = reward 
		else: target = self.temporal_difference(reward, new_state, done) 
		mse = tf.math.square(predicted_value - target)

		#
		return tf.math.reduce_mean(mse)


	def get_critic_model( self ):

		"""
		method that returns the neural network to train for the critic

		Returns:
		--------
			critic : keras.Model
				the default keras model for the critic
		"""

		# Generate the standard network using keras/tf
		inputs = keras.layers.Input(shape=self.input_shape)
		hidden_0 = keras.layers.Dense(64, activation='relu')(inputs)
		hidden_1 = keras.layers.Dense(64, activation='relu')(hidden_0)
		outputs = keras.layers.Dense(1, activation='linear')(hidden_1)

		#
		return keras.Model(inputs, outputs)


	@abc.abstractmethod
	def plot_results( self, episode, ep_reward, ep_reward_mean, reward_list ): 
		
		"""
		abstract method that select the action given the state, calling the actor network
		or the expolaration policy

		Parameters
		----------
			episode : int
				current running episode
			ep_reward : int
				reward of the current episode
			ep_reward_mean : queue
				reward list of the last 100 episodes, useful to compute the mean value of the last 100 episodes
			reward_list : list
				complete list of the rewards
		"""

		#
		return


	@abc.abstractmethod
	def get_action( self, state ): 

		"""
		abstract method that select the action given the state, calling the actor network
		or the expolaration policy

		Parameters
		----------
			state : NumpyAarray 
				that represent the current state to use to compute the next action

		Returns:
		--------
			action : int/float
				the action to perform, an integer index for discrete action space
				or a float value (or list of float) for continuous action space
		"""
		
		#
		return


	@abc.abstractmethod
	def actor_objective_function( self, memory_buffer ): 
		"""
		abstract method that compute the objective function for the actor, returns the objective function to optimize

		Parameters
		----------
			memory_buffer : NumpyArray 
				buffer with the data from each episode, in the following form:
				[state, action, action_prob, reward, new_state, done]

		Returns:
		--------
			objective_function : tf.EagerTensor
				tensorflow tensor that represent the objective function to maximize
		"""
		
		#
		return


	@abc.abstractmethod
	def get_actor_model( self ): 
				
		"""
		abstract method that returns the neural network to train for the actor

		Returns:
		--------
			critic : keras.Model
				the default keras model for the critic
		"""
		#
		return
