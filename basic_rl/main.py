from basic_rl.algorithms.REINFORCE import Reinforce
from basic_rl.algorithms.PPO import PPO
from basic_rl.algorithms.mcPPO import MonteCarloPPO
from basic_rl.algorithms.DDQN import DDQN
from basic_rl.algorithms.DDPG import DDPG
import gym

class BasicRL:

	discrete_dictionary = {
		"REINFORCE" : Reinforce,
		"PPO" : PPO, 
		"mcPPO" : MonteCarloPPO,
		"DDQN" : DDQN
	}

	continuous_dictionary = {
		"DDPG" : DDPG
	}

	def __init__(self, algo, gym_env, **kwargs):

		self.discrete_env = (type(gym_env.action_space) == gym.spaces.discrete.Discrete)

		if self.discrete_env and algo not in self.discrete_dictionary.keys(): 
			raise ValueError( f"Invalid Algorithm for Discrete Environment (valid keys: {list(self.discrete_dictionary.keys())})" )

		if not self.discrete_env and algo not in self.continuous_dictionary.keys(): 
			raise ValueError( f"Invalid Key for Discrete Environment (valid keys: {list(self.continuous_dictionary.keys())})" )

		kwargs["name"] = algo

		if self.discrete_env:
			self.algorithm = self.discrete_dictionary[algo]( gym_env, **kwargs )
		else: 
			self.algorithm = self.continuous_dictionary[algo]( gym_env, **kwargs )


	def train( self, num_episode=1000 ):
		self.algorithm.loop( num_episode )

