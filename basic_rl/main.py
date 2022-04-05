from basic_rl.algorithms.REINFORCE import Reinforce
from basic_rl.algorithms.PPO import PPO, ContPPO
from basic_rl.algorithms.mcPPO import MonteCarloPPO, ContMonteCarloPPO
from basic_rl.algorithms.DDQN import DDQN
from basic_rl.algorithms.DDPG import DDPG
import gym


class BasicRL:

	"""
	Main class of the NetVer project, this project implements different methods for the formal verificaiton of neural network. Currently the support is 
	only for feed forward neural netowrk and a proeprty can be expressed in three different format, see https://github.com/d-corsi/NetworkVerifier for 
	a complete list of the supported format and algorithms.

	Attributes
	----------
		algo : string
			the requested trianing algorithm to use for the training, a complete list of the supported algorithm at:
			https://github.com/d-corsi/BasicRL
		gym_env: dict
			the environment for the training in the OpenAI Gym format
		**kwargs: dict
			a dictionary to change the default paramters for the algorithm, a complete list can be found at:
			https://github.com/d-corsi/BasicRL

		
	Methods
	-------
		train( num_episode )
			method to start the training, the parameter num_episode (integer) indicates the number of epoch for the 
			reinforcement learning loop

	"""


	# Dictionary for the algorithms that support discrete environments
	# Translates the string into the class object
	discrete_dictionary = {
		"REINFORCE" : Reinforce,
		"PPO" : PPO, 
		"mcPPO" : MonteCarloPPO,
		"DDQN" : DDQN
	}

	# Dictionary for the algorithms that support continuous environments
	# Translates the string into the class object
	continuous_dictionary = {
		"DDPG" : DDPG,
		"PPO" : ContPPO,
		"mcPPO" : ContMonteCarloPPO
	}


	# Constructor of the class
	def __init__(self, algo, gym_env, **kwargs):

		# Chceck if the environment is discrete or continuous based on the observation space
		# of the gym enviornment
		self.discrete_env = (type(gym_env.action_space) == gym.spaces.discrete.Discrete)

		# Check if the requested algorithm is compatible with the provided environment
		if self.discrete_env and algo not in self.discrete_dictionary.keys(): 
			raise ValueError( f"Invalid Algorithm for Discrete Environment (valid keys: {list(self.discrete_dictionary.keys())})" )

		# Check if the requested algorithm is compatible with the provided environment
		if not self.discrete_env and algo not in self.continuous_dictionary.keys(): 
			raise ValueError( f"Invalid Key for Discrete Environment (valid keys: {list(self.continuous_dictionary.keys())})" )

		# Assign to the variable algorithm the reqeusted algorithm, from the discrete or continuous dictionary
		if self.discrete_env: self.algorithm = self.discrete_dictionary[algo]( gym_env, **kwargs )
		else: self.algorithm = self.continuous_dictionary[algo]( gym_env, **kwargs )


	# Method that run the training
	def train( self, num_episode=1000 ):

		# Calling the method "loop" of the reqeusted algorithm class
		self.algorithm.loop( num_episode )

