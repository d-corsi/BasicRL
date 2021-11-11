from basic_rl.algorithms.REINFORCE import ContinuousREINFORCE, DiscreteREINFORCE
from basic_rl.algorithms.ActorCritic import ContinuousActorCritic, DiscreteActorCritic
from basic_rl.algorithms.A2C import ContinuousA2C, DiscreteA2C
from basic_rl.algorithms.PPO import ContinuousPPO, DiscretePPO
from basic_rl.algorithms.mcPPO import ContinuousMcPPO, DiscreteMcPPO
from basic_rl.algorithms.DQN import DDQN
from basic_rl.algorithms.DDPG import DDPG
from basic_rl.algorithms.TD3 import TD3
import gym

class BasicRL:

	"""
	Main class of BasicRL. This class provide an interface to easly run different reinforcement learning
	algorithm on the Open AI Gym environments. The implementation currently provide the following algorithms:

		[x] REINFORCE
		[x] Actor/Critic
		[x] A2C
		[x] PPO 
		[x] PPO (montecarlo)
		[x] DDPG
		[ ] C51
		[x] TD3
		[ ] SAC
		[x] DDQN

	The constructor takes as input a dictionary to change all the parameters for the training, the complete parameter list
	and an example can be viewed on the GitHub page: https://github.com/d-corsi/BasicRL. 
	
	Attributes
	----------
		algorithm : BasicReinforcement
			a string that represent the variable of the bucket (literals)

		discrete_dictionary : dict
			dictionary that translate the discrete algorith key to the corresponding class

		continuous_dictionary : dict
			dictionary that translate the continuous algorith key to the corresponding class 
		
	Methods
	-------
		train( episodes )
			method that start the training loop with the given parameters
	"""

	# Dictionary from key to discrete algorithm class
	discrete_dictionary = {
		"REINFORCE" : DiscreteREINFORCE, 
		"ActorCritic" : DiscreteActorCritic, 
		"A2C" : DiscreteA2C, 
		"PPO" : DiscretePPO, 
		"mcPPO" : DiscreteMcPPO, 
		"DQN" : DDQN
	}

	# Dictionary from key to discrete algorithm class
	continuous_dictionary = {
		"REINFORCE" : ContinuousREINFORCE, 
		"ActorCritic" : ContinuousActorCritic, 
		"A2C" : ContinuousA2C, 
		"PPO" : ContinuousPPO, 
		"mcPPO" : ContinuousMcPPO, 
		"DDPG" : DDPG, 
		"TD3" : TD3
	}

	def __init__(self, algo, gym_env, **kwargs):

		"""
		Constructor of the class

		Parameters
		----------
			algo : str
				a string that represent the variable of the bucket (literals)
			gym_env : Gym
				the environment to train, in the Open AI Gym style 
			kwargs : dict
				dictionary to update all the parameters for the training in the python style, it is possibile to override
				every paramter of the training [e.g., basic_rl = BasicRL( "DQN", gym_env, gamma=0.85, verbose=2 )]
		"""

		# Use the action space of the given environment to find if it is continuous or discrete
		discrete_env = (type(gym_env.action_space) == gym.spaces.discrete.Discrete)

		# Ensure that the select algorithm is (i) implemented and (ii) compatible with the given environment (discrete check)
		if discrete_env and algo not in self.discrete_dictionary.keys(): 
			raise ValueError( f"Invalid Algorithm for Discrete Environment (valid keys: {list(self.discrete_dictionary.keys())})" )

		# Ensure that the select algorithm is (i) implemented and (ii) compatible with the given environment (continuous check)
		if not discrete_env and algo not in self.continuous_dictionary.keys(): 
			raise ValueError( f"Invalid Key for Discrete Environment (valid keys: {list(self.continuous_dictionary.keys())})" )

		# Add the algorithm name to the argument for BasicReinforcement, necessary for printing and saving purpose
		kwargs["name"] = algo

		# Select the algorithm from the dictionary with the given key and coll the constructor
		self.algorithm = self.discrete_dictionary[algo] if discrete_env else self.continuous_dictionary[algo]
		self.algorithm = self.discrete_dictionary[algo]( gym_env, **kwargs )


	def train( self, episodes=1000 ):

		"""
		Start the main training loop of the selected algorithm on the environment,
		(call the method "learn" of BasicReinforcement).

		Parameters
		----------
			episodes : int
				number of episodes for the training
		"""

		# Add the number of episodes for the training and call the learn method
		self.algorithm.num_episodes = episodes
		self.algorithm.learn()

