import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import abc, csv

class ReinforcementLearning( metaclass=abc.ABCMeta ):


	def __init__( self, env, verbose, str_mod, seed ):

			if seed is None: seed = np.random.randint(0, 1000)

			tf.random.set_seed( seed )
			np.random.seed( seed )

			self.env = env
			self.verbose = verbose
			self.run_id = seed
			self.str_mod = str_mod

			self.input_shape = self.env.observation_space.shape
			self.action_space = env.action_space


	def loop( self, num_episodes=10000 ):		

		# Initialize the loggers
		logger_dict = { "reward": [], "step": [] }

		# Setup the environment for for the I/O on file (logger/models)
		# (only when verbose is active on file)
		if self.verbose > 1:

			# Create the string with the configuration for the file name
			file_name = f"{self.str_mod}_seed{self.run_id}"

			#
			for key, value in self.relevant_params.items():
				file_name += f"_{value}{self.__dict__[key]}"

			# Create the CSV file and the writer
			csv_file = open( f"data/{file_name}.csv", mode='w')
			fieldnames = ['episode', 'reward', 'step']
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

			# Log all the results, depending on the <verbose> parameter
			# here simple print of the results
			if self.verbose > 0:
				last_n =  min(len(logger_dict['reward']), 100)
				reward_last_100 = logger_dict['reward'][-last_n:]
				step_last_100 = logger_dict['step'][-last_n:]

				print( f"({self.str_mod}) Ep: {episode:5}", end=" " )
				print( f"reward: {logger_dict['reward'][-1]:7.2f} (last_100: {np.mean(reward_last_100):7.2f})", end=" " )
				if 'eps_greedy' in self.__dict__.keys(): print( f"eps: {self.eps_greedy:3.2f}", end=" " )
				print( f"step_last_100 {int(np.mean(step_last_100)):5d}")

			# Log all the results, depending on the <verbose> parameter
			# here save the log to a CSV file
			if self.verbose > 1:	
				self.writer.writerow({ 
					'episode' : episode,
					'reward': logger_dict['reward'][-1],  
					'step': logger_dict['step'][-1]	
				})


	def generate_model( self, input_shape, output_size=1, layers=2, nodes=32, last_activation='linear' ):

		#
		hiddens_layers = [tf.keras.layers.Input( shape=input_shape )]
		for _ in range(layers):	hiddens_layers.append( tf.keras.layers.Dense( nodes, activation='relu')( hiddens_layers[-1] ) )
		hiddens_layers.append( tf.keras.layers.Dense( output_size, activation=last_activation)( hiddens_layers[-1] ) )	

		#
		return tf.keras.Model( hiddens_layers[0], hiddens_layers[-1] )