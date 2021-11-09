from basic_rl.algorithms.REINFORCE import ContinuousREINFORCE, DiscreteREINFORCE
from basic_rl.algorithms.ActorCritic import ContinuousActorCritic, DiscreteActorCritic
from basic_rl.algorithms.A2C import ContinuousA2C, DiscreteA2C
from basic_rl.algorithms.PPO import ContinuousPPO, DiscretePPO
from basic_rl.algorithms.mcPPO import ContinuousMcPPO, DiscreteMcPPO
from basic_rl.algorithms.DQN import DDQN
from basic_rl.algorithms.DDPG import DDPG
from basic_rl.algorithms.TD3 import TD3

class BasicRL:

	def __init__(self):

		import gym

		########################
		## Continuous Session ##
		########################

		gym_env = gym.make("LunarLanderContinuous-v2")

		algorithm = DDPG( gym_env, name="DDPG", num_episodes=2500, verbose=2 )
		algorithm.learn()

		algorithm = TD3( gym_env, name="TD3", num_episodes=2500, verbose=2 )
		algorithm.learn()

		algorithm = ContinuousREINFORCE( gym_env, name="ContinuousREINFORCE", num_episodes=2500, verbose=2 )
		algorithm.learn()

		algorithm = ContinuousActorCritic( gym_env, name="ContinuousActorCritic", num_episodes=2500, verbose=2 )
		algorithm.learn()

		algorithm = ContinuousA2C( gym_env, name="ContinuousA2C", num_episodes=2500, verbose=2 )
		algorithm.learn()

		algorithm = ContinuousPPO( gym_env, name="ContinuousPPO", num_episodes=2500, verbose=2 )
		algorithm.learn()
				
		algorithm = ContinuousMcPPO( gym_env, name="ContinuousMcPPO", num_episodes=2500, verbose=2 )
		algorithm.learn()
		
		########################
		##  Discrete Session  ##
		########################

		gym_env = gym.make("LunarLander-v2")

		algorithm = DDQN( gym_env, name="DDQN", num_episodes=2500, verbose=2 )
		algorithm.learn()
		
		algorithm = DiscreteREINFORCE( gym_env, name="DiscreteREINFORCE", num_episodes=2500, verbose=2 )
		algorithm.learn()

		algorithm = DiscreteActorCritic( gym_env, name="DiscreteActorCritic", num_episodes=2500, verbose=2 )
		algorithm.learn()

		algorithm = DiscreteA2C( gym_env, name="DiscreteA2C", num_episodes=2500, verbose=2 )
		algorithm.learn()

		algorithm = DiscretePPO( gym_env, name="DiscretePPO", num_episodes=2500, verbose=2 )
		algorithm.learn()
		
		algorithm = DiscreteMcPPO( gym_env, name="DiscreteMcPPO", num_episodes=2500, verbose=2 )
		algorithm.learn()
		