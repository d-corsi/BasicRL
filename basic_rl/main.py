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

	discrete_dictionary = {
		"REINFORCE" : DiscreteREINFORCE, 
		"ActorCritic" : DiscreteActorCritic, 
		"A2C" : DiscreteA2C, 
		"PPO" : DiscretePPO, 
		"mcPPO" : DiscreteMcPPO, 
		"DQN" : DDQN
	}

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


	def train( self ):
		self.algorithm.learn()

