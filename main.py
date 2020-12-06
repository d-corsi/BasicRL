import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from BasicRL.BasicRL import BasicRL
import gym; gym.logger.set_level(40)


if __name__ == "__main__":
	print("Hello Basic RL!")

	#env = gym.make("CartPole-v1")
	#env = gym.make("LunarLanderContinuous-v2")
	#env = gym.make("LunarLander-v2")
	from AquaSmallEnv import AquaSmallEnv; env = AquaSmallEnv( with_obstacles=False )

	learner = BasicRL("mcPPO", gym_env=env, verbose=2)
	learner.learn(5000)
