import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import abc, csv, datetime, gym


# Utility function, check existence of a folder
def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)


class ReinforcementLearning( metaclass=abc.ABCMeta ):

	"""
	Abstract class that implements the main reinforcement learning loop. This class is responsible to interact with the environment,
	performing the action and obtaining the information from the Gym object. It also implements a logging system that save the reward and
	the step size on a 'csv' file. Finally, tt implements a generator of neural networks based on a set of parameters.

	Inside the main loop, at each step the class save inside the memory buffer the following informations:
		- state: the current state of the environment
		- action: the selected action for the current state
		- action_prob: the probability of the action (useful in gradient based method, None otherwise)
		- reward: the reward signal for the state/action pair returned by the envirnoment
		- new_state: the new state computed by the environment after the action
		- done: true if the new state is terminal state

	For classes that inherit from this, you must implements the following methods:
		- network_update_rule( episode, terminal ): implements the update rule for the network of the specific algorithm. As a paramters
			the method requires the current episode and a flag to indicate if it is a terminal state (end of the trajectory).
		- get_action( state ): called at each time step to select the action based on the given state. The method
			must returns two variable, the selected action and the corresponding probability (None in the deterministic case)

	"""


	# Constructor of the class
	def __init__( self, env, verbose, str_mod, seed=None ):

			# If the provided seed is None, geenrate a random seed
			if seed is None: seed = np.random.randint(0, 1000)

			# Set up of the random seed for tensorflow and numpy
			tf.random.set_seed( seed )
			np.random.seed( seed )

			# Assignment of the basic parameters the algorithm specific paramters 
			# must be setted in the inherit class
			self.env = env
			self.verbose = verbose
			self.run_id = seed
			self.str_mod = str_mod

			# Get input and output shape for the network from the enviornment
			# input/output shape
			self.input_shape = self.env.observation_space.shape
			self.action_space = env.action_space

			# Utility variable for data logging
			self.logging_dir_path = os.getcwd() + '/log'


	# Main loop of the algorithm, the parameter num_episodes indicates the number of reinforcement learning
	# episode for the training
	def loop( self, num_episodes=10000 ):		

		# Initialize the loggers
		logger_dict = { "reward": [], "step": [] }

		# Setup the environment for for the I/O on file (logger/models)
		# (only when verbose is active on file)
		if self.verbose > 1:

			# Create the string with the configuration for the file name
			self.logging_dir_path += f"/{self.str_mod}_seed{self.run_id}"

			# Extract from the dicitionary of the relevant parameters
			# the parameters to save on the filen name
			for key, value in self.relevant_params.items():
				self.logging_dir_path += f"_{value}{self.__dict__[key]}"

			# Adding the starting datetime for the file name
			self.logging_dir_path += '_'
			self.logging_dir_path += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

			# Saving the stats in the correct foler
			file_name = self.logging_dir_path + '/run_stats.csv'
			ensure_dir(file_name)

			# Create the CSV file and the writer
			csv_file = open(f"{file_name}", mode='x')
			fieldnames = ['episode', 'reward', 'success', 'step', 'cost', 'action_list']
			self.writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
			self.writer.writeheader()

		# Iterate the training loop over multiple episodes
		for episode in range(num_episodes):

			# Reset the environment at each new episode
			state = self.env.reset()

			# Initialize the values for the logger
			logger_dict['reward'].append(0)
			logger_dict['step'].append(0)

			# Main loop of the current episode
			while True:
				
				# Select the action, perform the action and save the returns in the memory buffer
				action, action_prob = self.get_action(state)
				new_state, reward, done, _ = self.env.step(action)
				self.memory_buffer.append([state, action, action_prob, reward, new_state, done])
				
				# Update the dictionaries for the logger and the trajectory
				logger_dict['reward'][-1] += reward	
				logger_dict['step'][-1] += 1	
				
				# Call the update rule of the algorithm
				self.network_update_rule( episode, done )

				# Exit if terminal state and eventually update the state
				if done: break
				state = new_state

			# Log all the results, depending on the <verbose> parameter.
			# To decide the paramenters to print we refer to the same attribute <relevant_params> described
			# for the file name
			if self.verbose > 0:
				last_n =  min(len(logger_dict['reward']), 100)
				reward_last_100 = logger_dict['reward'][-last_n:]
				step_last_100 = logger_dict['step'][-last_n:]

				# Actual print of the log, the eps_greed paramters is only for the deterministic policy
				# which implements an epsilon greedy strategy
				print( f"({self.str_mod}) Ep: {episode:5}", end=" " )
				print( f"reward: {logger_dict['reward'][-1]:7.2f} (last_100: {np.mean(reward_last_100):7.2f})", end=" " )
				if 'eps_greedy' in self.__dict__.keys(): print( f"eps: {self.eps_greedy:3.2f}", end=" " )
				if 'sigma' in self.__dict__.keys(): print( f"sigma: {self.sigma:3.2f}", end=" " )
				print( f"step_last_100 {int(np.mean(step_last_100)):5d}")

			# Log all the results, depending on the <verbose> parameter
			# here save the log to a CSV file
			if self.verbose > 1:	
				self.writer.writerow({ 
					'episode' : episode,
					'reward': logger_dict['reward'][-1],  
					'step': logger_dict['step'][-1]	
				})

			# Log all the results, depending on the <verbose> parameter
			# here save the models generated
			if self.verbose > 2 and episode % 100 == 0:
				saved_model_path = self.logging_dir_path + '/models'
				ensure_dir(f"{saved_model_path}/test.txt")
				if 'actor' in self.__dict__.keys():
					self.actor.save(f"{saved_model_path}/{self.str_mod}_id{self.run_id}_ep{episode}.h5")
				if 'network' in self.__dict__.keys(): 
					self.network.save(f"{saved_model_path}/{self.str_mod}_id{self.run_id}_ep{episode}.h5")


	# Class that generate a basic neural netowrk from the given parameters.
	# Can be overrided in the inheriting class for a specific architecture (e.g., dueling)
	def generate_model( self, input_shape, output_size=1, layers=2, nodes=32, last_activation='linear', output_bounds=None ):

		# Fix if the output shape is received as a gym spaces,
		# covnersion to integer
		if isinstance(output_size, gym.spaces.discrete.Discrete): output_size = output_size.n
		if isinstance(output_size, gym.spaces.box.Box): output_size = output_size.shape[0]		

		# Itearte over the provided parametrs to create the network, the input shape must be defined as a multidimensional tuple (e.g, (4,))
		# While the output size as an integer
		hiddens_layers = [tf.keras.layers.Input( shape=input_shape )]
		for _ in range(layers):	hiddens_layers.append( tf.keras.layers.Dense( nodes, activation='relu')( hiddens_layers[-1] ) )
		hiddens_layers.append( tf.keras.layers.Dense( output_size, activation=last_activation)( hiddens_layers[-1] ) )	

		# Normalize the output layer between a range if given,
		# usually used with continuous control for a sigmoid final activation
		if output_bounds is not None: 
			hiddens_layers[-1] = hiddens_layers[-1] * (output_bounds[1] - output_bounds[0]) + output_bounds[0]

		# Create the model with the keras format and return
		return tf.keras.Model( hiddens_layers[0], hiddens_layers[-1] )